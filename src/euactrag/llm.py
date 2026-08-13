"""One thin LLM client for every provider.

Groq, OpenAI, Gemini and Ollama all expose an OpenAI-compatible
`/chat/completions` endpoint, so a single HTTP call covers all four. That is why
there are no vendor SDKs in requirements.txt -- swapping providers is a change of
two environment variables, not a code change.

`extractive` is the no-key fallback: the pipeline degrades to returning ranked
source passages without a generated answer, so the demo never hard-fails.
"""
from __future__ import annotations

import os

import httpx

from . import config


class LLMUnavailable(RuntimeError):
    """No usable credentials for the selected provider."""


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

    payload = {
        "model": model or config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens or config.LLM_MAX_TOKENS,
    }
    r = httpx.post(
        f"{base}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
