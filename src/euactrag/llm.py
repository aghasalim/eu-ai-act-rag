"""One thin LLM client for every provider.

Groq, OpenAI, Gemini and Ollama all expose an OpenAI-compatible
`/chat/completions` endpoint, so a single HTTP call covers all four. That is why
there are no vendor SDKs in requirements.txt, swapping providers is a change of
two environment variables, not a code change.

`extractive` is the no-key fallback: the pipeline degrades to returning ranked
source passages without a generated answer, so the demo never hard-fails.
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict, deque

import httpx

from . import config

MAX_RETRIES = 8

_DURATION = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?$")

# --- client-side tokens-per-minute pacing --------------------------------
# Free tiers meter tokens per minute (Groq: 12k for llama-3.3-70b, 8k for the
# gpt-oss-120b judge). The provider refills continuously: reset-tokens reads
# ~205ms: so `x-ratelimit-remaining-tokens` is almost always near full and
# reacting to it never fires, yet a burst still trips a 429. A full evaluation
# is ~145 calls carrying ~3k tokens of retrieved context each, so it must pace
# itself. We learn each model's ceiling from the first response header and spend
# at most USAGE_FRACTION of it.
_WINDOW = 60.0
USAGE_FRACTION = float(os.getenv("LLM_USAGE_FRACTION", "0.75"))
_DEFAULT_TPM = int(os.getenv("LLM_TPM_BUDGET", "8000"))
_limits: dict[str, int] = {}
_spent: dict[str, deque] = defaultdict(deque)


def _budget(model: str) -> float:
    return _limits.get(model, _DEFAULT_TPM) * USAGE_FRACTION


def _reserve(model: str, want: int) -> None:
    """Block until `want` tokens fit inside this model's rolling minute."""
    dq = _spent[model]
    while True:
        now = time.monotonic()
        while dq and now - dq[0][0] > _WINDOW:
            dq.popleft()
        if not dq or sum(t for _, t in dq) + want <= _budget(model):
            return
        time.sleep(min(_WINDOW - (now - dq[0][0]) + 0.25, 65.0))


def _estimate(messages: list[dict], max_tokens: int) -> int:
    chars = sum(len(str(m.get("content", ""))) for m in messages)
    return int(chars / 3.5) + max_tokens


def parse_duration(s: str) -> float:
    """Parse the provider's rate-limit reset format: '205ms', '7.66s', '59m2.4s'."""
    s = (s or "").strip()
    if not s:
        return 0.0
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except ValueError:
            return 0.0
    m = _DURATION.match(s)
    if not m:
        return 0.0
    h, mi, sec = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(sec or 0)


class LLMUnavailable(RuntimeError):
    """No usable credentials for the selected provider."""


class DailyQuotaExhausted(RuntimeError):
    """The per-day token allowance for this model is spent.

    Distinct from a per-minute limit because it is not waitable in-process: the
    window is hours long. Raised immediately so a caller can checkpoint and stop
    rather than burning its retry budget on sleeps that cannot possibly succeed.
    """


PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "ollama": (os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1", None),
}


def available(provider: str | None = None) -> bool:
    provider = provider or config.LLM_PROVIDER
    if provider == "extractive":
        return False
    if provider not in PROVIDERS:
        return False
    _, key_env = PROVIDERS[provider]
    return key_env is None or bool(os.getenv(key_env))


def chat(
    messages: list[dict],
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 90.0,
    reasoning_effort: str | None = None,
) -> str:
    provider = provider or config.LLM_PROVIDER
    if provider not in PROVIDERS:
        raise LLMUnavailable(f"unknown provider {provider!r}")
    base, key_env = PROVIDERS[provider]
    api_key = os.getenv(key_env) if key_env else "ollama"
    if key_env and not api_key:
        raise LLMUnavailable(
            f"{key_env} is not set. Put it in .env, or set LLM_PROVIDER=extractive "
            f"to run retrieval-only."
        )

    mdl = model or config.LLM_MODEL
    cap = max_tokens or config.LLM_MAX_TOKENS
    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": cap,
    }
    # Reasoning models spend the max_tokens budget on a <think> block before
    # emitting anything, so a judge asked for strict JSON gets truncated mid
    # thought and returns nothing parseable. Turning reasoning off makes the
    # reply deterministic-shaped and ~10x cheaper, which matters against a
    # per-day token allowance.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    # Wait for room in this model's rolling minute before spending it, then keep
    # the 429 retry as a backstop for whatever the estimate gets wrong.
    est = _estimate(messages, cap)
    _reserve(mdl, est)

    delay = 2.0
    for attempt in range(MAX_RETRIES):
        r = httpx.post(
            f"{base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        try:
            _limits[mdl] = int(r.headers["x-ratelimit-limit-tokens"])
        except (KeyError, ValueError):
            pass

        if r.status_code == 429:
            # A per-day limit is not a per-minute limit. Groq only reveals it in
            # the error body: the x-ratelimit-* headers describe RPM/TPM only,
            # so nothing else here can see it coming. Retrying is futile.
            body = r.text
            if "per day" in body or "TPD" in body:
                m = re.search(r"try again in ([0-9hms.]+)", body)
                mins = parse_duration(m.group(1)) / 60 if m else 0.0
                raise DailyQuotaExhausted(
                    f"{mdl}: daily token quota spent"
                    + (f"; resets in ~{mins:.0f} min" if mins else "")
                    + f". {body[:200]}"
                )

        if r.status_code == 429 or r.status_code >= 500:
            if attempt == MAX_RETRIES - 1:
                break
            # Charge the estimate anyway: the request consumed budget even though
            # it was refused, so the pacer must not think the minute is free.
            _spent[mdl].append((time.monotonic(), est))
            wait = float(r.headers.get("retry-after") or 0) or delay
            time.sleep(min(wait + 0.5, 120.0))
            delay *= 2
            continue

        r.raise_for_status()
        body = r.json()
        used = (body.get("usage") or {}).get("total_tokens") or est
        _spent[mdl].append((time.monotonic(), int(used)))
        return body["choices"][0]["message"]["content"].strip()

    r.raise_for_status()  # out of retries: surface the real error
    return r.json()["choices"][0]["message"]["content"].strip()
