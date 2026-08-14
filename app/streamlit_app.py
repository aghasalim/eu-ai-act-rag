"""Streamlit demo. The point of the layout is that the retrieved evidence sits
next to the answer, so a reader can check the answer instead of trusting it."""
from __future__ import annotations

import sys
from pathlib import Path

# Must happen before anything imports chromadb. Streamlit Community Cloud runs a
# Debian image whose system sqlite3 predates 3.35, which chromadb refuses to
# start on. pysqlite3-binary ships a modern build; we swap it in under the
# stdlib name. Linux-only -- macOS and Windows sqlite3 are new enough.
try:  # noqa: SIM105
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import json
import os

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv()

# Streamlit Community Cloud delivers secrets through st.secrets, but everything
# below app/ reads os.getenv so it stays host-agnostic (a .env locally, real env
# vars in Docker). Bridge the two here, and do it *before* importing config,
# which snapshots these values at import time. Accessing st.secrets raises when
# no secrets are configured at all, hence the guard.
for _key in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
             "LLM_PROVIDER", "LLM_MODEL", "RETRIEVAL_MODE", "TOP_K"):
    try:
        if not os.getenv(_key) and _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        break

from src.euactrag import config, llm, pipeline  # noqa: E402

st.set_page_config(page_title="EU AI Act RAG", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading index and embedding model (first run "
                                "builds the index, ~2 min)...")
def warm():
    """Load the index, building it first if this host has none.

    `data/index/` is deliberately not in git -- a committed Chroma store is a
    binary tied to one chromadb version, and it silently rots on upgrade. The
    chunks are committed instead, so any host can rebuild the index from them.
    On a platform like Streamlit Community Cloud that clones the repo and runs
    the app directly, this is the difference between working and crashing on
    first request.
    """
    from src.euactrag import index as idx, retrieve

    try:
        idx.get_collection().count()
    except Exception:
        idx.build()
    retrieve.search("warmup", k=1, mode="hybrid")
    return True


EXAMPLES = [
    "What is the maximum fine for using a prohibited AI practice?",
    "Is emotion recognition in the workplace allowed?",
    "When do the rules for general-purpose AI models start to apply?",
    "Is an AI system used to filter job applications high-risk?",
    "What are the six lawful bases for processing data under the GDPR?",
]

st.title("⚖️ EU AI Act — Retrieval-Augmented QA")
st.caption(
    "Answers are grounded in Regulation (EU) 2024/1689 only. Every claim is cited, "
    "and the passages the answer was built from are shown on the right. "
    "The last example question is deliberately out of scope — the system should "
    "refuse it."
)

with st.sidebar:
    st.header("Retrieval")
    mode = st.radio("Strategy", ["hybrid", "dense", "bm25"], index=0,
                    help="hybrid = dense + BM25 fused with reciprocal rank fusion")
    k = st.slider("Passages retrieved (k)", 3, 12, config.TOP_K)
    st.divider()
    st.header("Generation")
    if llm.available():
        st.success(f"{config.LLM_PROVIDER} · {config.LLM_MODEL}")
    else:
        st.warning(
            "No API key set — running in retrieval-only mode. Retrieval below is "
            "fully live; only the written answer is disabled. To enable it set "
            "`GROQ_API_KEY` — in **app secrets** when hosted, or in `.env` locally."
        )
    st.divider()
    st.markdown(
        "**Measured, not asserted.** The scores below come from "
        "[`eval/results/eval_latest.json`](https://github.com/aghasalim/eu-ai-act-rag/tree/main/eval/results), "
        "the same file `make eval` writes — nothing here is typed in by hand."
    )

@st.cache_data
def scores() -> dict | None:
    """The measured evaluation, read from disk rather than written into the page.

    Every metric shown below is looked up from this file. That is deliberate: the
    README of this project drifted from the generated results twice (an MRR cell
    and an nDCG cell), and a demo that hardcodes its own numbers is the same bug
    with a wider audience.
    """
    p = Path(__file__).resolve().parents[1] / "eval" / "results" / "eval_latest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


_ev = scores()
if _ev:
    cfg, summ = _ev["config"], _ev["summary"]
    hyb = _ev["retrieval"]["hybrid"]["at_k"].get(str(cfg["top_k"]), {}).get("overall", {})
    with st.expander(
        f"How well does this actually work? — measured on {cfg['n_questions']} "
        f"hand-written questions", expanded=False,
    ):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faithfulness", f"{summ['faithfulness']:.1%}",
                  help="Share of generated claims entailed by the retrieved passages.")
        c2.metric("Citation validity", f"{summ['citation_validity']:.0%}",
                  help="Citations that point at a passage actually retrieved.")
        c3.metric("Refused out-of-scope", f"{summ['correct_abstention_rate']:.0%}",
                  f"{summ['hallucination_rate_unanswerable']:.0%} hallucinated")
        c4.metric("Retrieval hit rate", f"{hyb.get('hit_rate', 0):.1%}",
                  f"full recall {hyb.get('full_recall', 0):.1%}")

        by = summ["by_type"]
        st.markdown(
            f"Answer accuracy is **{summ['answer_accuracy_strict']:.1%}** strict "
            f"(**{summ['answer_accuracy_lenient']:.1%}** counting partial credit), and it "
            f"splits hard by question type: **{by['single_hop']['accuracy_strict']:.1%}** on "
            f"single-hop questions against **{by['multi_hop']['accuracy_strict']:.1%}** on "
            f"multi-hop ones that need two articles at once. It refuses "
            f"{summ['false_abstention_rate']:.1%} of questions it could have answered — "
            "erring toward refusal, which for a legal assistant is the right direction "
            "to err in.\n\n"
            f"Answered by `{cfg['gen_model']}`, graded by `{cfg['judge_model']}` — a "
            "different model family, so it is not marking its own work."
        )

    # The running configuration can differ from the evaluated one -- via the
    # sidebar, or because this host answers with a different model than the eval
    # used. Either way the numbers above stop describing what the visitor is
    # getting, and a score that silently borrows authority from a different setup
    # is the exact failure this project exists to argue against.
    drift = []
    if mode != cfg["gen_mode"] or k != cfg["top_k"]:
        drift.append(f"retrieval is **{mode}, k={k}** here vs **{cfg['gen_mode']}, "
                     f"k={cfg['top_k']}** when measured")
    if llm.available() and config.LLM_MODEL != cfg["gen_model"]:
        drift.append(f"answers come from **{config.LLM_MODEL}** here vs "
                     f"**{cfg['gen_model']}** when measured")
    if drift:
        st.caption("⚠️ " + "; ".join(drift)
                   + ". The scores above do not describe this configuration.")

warm()

if "q" not in st.session_state:
    st.session_state.q = ""
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    if c.button(ex[:28] + "…", help=ex, use_container_width=True):
        st.session_state.q = ex

question = st.text_input(
    "Ask a question about the EU AI Act",
    value=st.session_state.q,
    placeholder="e.g. What compute threshold makes a GPAI model systemic risk?",
)

if question:
    with st.spinner("Retrieving and generating…"):
        res = pipeline.answer(question, k=k, mode=mode)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Answer")
        if res.abstained:
            st.warning(
                "**Not answerable from the corpus.** The system declined rather "
                "than guessing — this is the intended behaviour for out-of-scope "
                "questions."
            )
        st.markdown(res.answer)
        if res.cited_units:
            retrieved = {c["unit_id"] for c in res.contexts}
            bad = [u for u in res.cited_units if u not in retrieved]
            if bad:
                st.error(f"Citations not present in retrieved context: {bad}")
            else:
                st.success(
                    f"All {len(res.cited_units)} citations trace to retrieved passages."
                )

    with right:
        st.subheader(f"Retrieved passages ({len(res.contexts)})")
        st.caption(f"strategy: `{res.mode}` · these are the only sources the model saw")
        for h in res.contexts:
            cited = h["unit_id"] in res.cited_units
            with st.expander(
                f"{'✅ ' if cited else ''}[{h['rank']}] {h['citation']}", expanded=cited
            ):
                st.caption(
                    f"{h['chapter_title'] or h['kind']} · score {h['score']:.4f} · "
                    f"`{h['chunk_id']}`"
                )
                body = h["text"].split("\n", 1)[-1]
                st.text(body[:2400] + ("…" if len(body) > 2400 else ""))
                st.markdown(f"[View on EUR-Lex]({h['url']})")

st.divider()
st.caption(
    "Corpus: Regulation (EU) 2024/1689, retrieved from the EU Publications Office "
    "(CELEX 32024R1689). Not legal advice."
)
