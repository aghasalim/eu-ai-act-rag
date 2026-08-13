# EU AI Act — RAG with a real evaluation harness

[![ci](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Grounded question answering over **Regulation (EU) 2024/1689 (the EU AI Act)**, where
the evaluation — not the demo — is the deliverable.

Most RAG portfolio projects stop at "it answers questions." This one measures
whether the answers are actually supported by the retrieved text, how often the
system correctly refuses a question it cannot answer, and — when it fails — *why*.

---

## Headline results

45 hand-written questions (21 single-hop, 12 multi-hop, 12 deliberately
unanswerable) over 464 chunks. Retrieval metrics are exact and require no LLM.
Full methodology and per-question results: **[RESULTS.md](RESULTS.md)**.

### Retrieval @ k=6

| strategy | hit rate | recall | full recall | MRR | nDCG | s/query |
|---|---|---|---|---|---|---|
| dense (bge-small) | 81.8% | 67.2% | 51.5% | 0.521 | 0.537 | 0.402 |
| BM25 | 84.9% | 73.2% | 63.6% | 0.561 | 0.571 | 0.002 |
| **hybrid (RRF)** | **90.9%** | **80.3%** | **69.7%** | **0.790** | **0.754** | 0.032 |

*hit rate* = at least one required provision retrieved. *full recall* = **every**
required provision retrieved — the metric that decides whether a multi-hop
question can be answered at all.

### Three findings worth stating plainly

**1. BM25 beats dense retrieval on this corpus.** Legal text is precise-terminology
heavy: "serious incident", "putting into service", "Annex III" are exact strings, and
a 384-dimensional embedding blurs precisely the distinctions the Regulation is built
on. Hybrid retrieval wins because the two methods fail on different questions, not
because dense is strong here.

**2. Multi-hop retrieval is the real bottleneck, and it is much worse than the
headline suggests.**

| strategy | multi-hop hit rate | multi-hop **full** recall |
|---|---|---|
| dense | 91.7% | 8.3% |
| BM25 | 91.7% | 33.3% |
| hybrid | 100.0% | 41.7% |

Hybrid retrieval finds *something* relevant for every multi-hop question, but finds
*everything* required for only 41.7% of them. A single averaged "recall" number hides
this completely. This is the failure mode that produces confidently half-right
answers — the rule without its exception, the obligation without its deadline.

**3. Recitals were poisoning retrieval.** The Act's 180 recitals restate the operative
rules in flowing prose, so they match a natural-language question *better* than the
terse article that actually contains the rule. Down-weighting them in rank fusion
moved MRR from 0.581 to 0.790 (+36%). The gain is a **step, not a peak** — every
weight below 1.0 scores identically — so it is a structural fix rather than a
constant fitted to this question set. See the ablation in
[RESULTS.md](RESULTS.md#ablation-down-weighting-recitals), including the caveat that
this eval set contains no recital-answerable questions and so overstates the benefit.

### Generation, faithfulness and abstention

Requires an API key (free tier is enough), then `make eval && make report` fills in
the answer-quality half of [RESULTS.md](RESULTS.md) — faithfulness, citation
validity, hallucination rate on the 12 unanswerable questions, and a categorised
failure table. **These numbers are not reported here until that run has been done on
your key**; the harness is built and tested, and this README will not carry a number
the repository cannot reproduce.

---

## Quickstart

```bash
make setup && make corpus && make index
```

```bash
make eval-retrieval
```

That reproduces every retrieval number above with **no API key and no network calls
to an LLM**. For generated answers and faithfulness scoring, add a free
[Groq](https://console.groq.com/keys) key:

```bash
cp .env.example .env && echo "GROQ_API_KEY=your_key_here" >> .env
```

```bash
make eval && make report && make app
```

---

## How it works

### Corpus and chunking

Source is the official XHTML from the EU Publications Office Cellar API
(CELEX `32024R1689`) — `eur-lex.europa.eu` returns a bot-challenge page to scripted
clients, while Cellar serves the same authoritative text with stable ELI markup.

**Chunks follow the document's own structure, not a fixed window.** A legal answer
*is* a citation ("Article 6(2)", "Annex III point 5(b)"), so the retrieval unit
should be the unit a lawyer would cite. A 512-token sliding window routinely severs
an article's scope clause from its exceptions paragraph — and in this Regulation the
exception is usually the answer.

That creates three problems, each handled explicitly:

| problem | handling |
|---|---|
| Articles range from 2 lines (Art. 4) to ~9k tokens (Art. 3, 68 definitions) | Split at enumerated-item boundaries, then sentence boundaries — never mid-sentence |
| A split chunk retrieved alone loses its identity ("…shall not apply" — *what* shall not?) | Every chunk carries a `Chapter > Section > Article N — Title` breadcrumb, embedded with the body |
| Enumerations are nested two-column HTML tables; `get_text()` turns them to soup | Recursive renderer rebuilds `1. … (a) …` outline structure |

Result: 113 articles + 180 recitals + 13 annexes → **464 chunks**, p95 = 440 tokens,
max = 468 — all inside the encoder's 512-token window. Anything above that would be
silently truncated at embedding time, so `tests/` asserts it can't happen.

The parser also repairs the `10^25` FLOP threshold in Article 51(2), which is marked
up as a superscript and flattens to a meaningless `"10 25"` under naive extraction.

### Retrieval

`BAAI/bge-small-en-v1.5` in Chroma (cosine) fused with BM25 via **Reciprocal Rank
Fusion**. RRF rather than a weighted score blend: cosine similarities and BM25 scores
live on incompatible scales, so blending needs a normalisation constant retuned per
corpus. RRF consumes only *ranks* — one parameter, nothing fitted to the eval set.

### Generation

Groq, OpenAI, Gemini and Ollama all expose an OpenAI-compatible `/chat/completions`
endpoint, so one `httpx` call covers all four and `requirements.txt` carries **no
vendor SDK**. Switching provider is two environment variables.

The prompt gives the model an explicit abstention token and tells it the corpus
boundary. Without an escape hatch an instruction-tuned model will answer an
out-of-scope question anyway — which is exactly what the 12 unanswerable questions
measure. With no API key the pipeline degrades to retrieval-only rather than
inventing an answer.

---

## The evaluation set

45 questions in [`eval/qa_set.jsonl`](eval/qa_set.jsonl), hand-written against text
verified in the parsed corpus, each carrying gold provision ids and a `probe` field
recording which failure it is designed to catch.

- **21 single-hop** — directly stated facts.
- **12 multi-hop** — require combining two or more provisions.
- **12 unanswerable** — GDPR, the Digital Services Act, institutional facts the
  Regulation never states. These measure hallucination, not knowledge.

Questions deliberately target places where a nearly-right answer is *wrong*:

- Article 73 has **three** competing serious-incident deadlines (15 days general,
  10 days on death, 2 days for widespread infringement).
- Article 99 caps fines at "whichever is **higher**" — but Article 99(6) inverts
  this to "whichever is **lower**" for SMEs.
- Article 2(8) exempts research and development, then exempts real-world testing
  *back out* of that exemption.

### Metrics

| metric | needs an LLM? | what it catches |
|---|---|---|
| hit rate / recall / **full recall** / MRR / nDCG | no | wrong or incomplete passages retrieved |
| **citation validity** | no | citations pointing at text that was never retrieved — fabricated attribution |
| faithfulness (per atomic claim) | yes | answer asserting more than the passages support |
| answer correctness vs reference | yes | right sources, wrong conclusion |
| correct-abstention / hallucination rate | yes | answering a question the corpus cannot answer |

Two guards against the usual criticism of LLM-graded evaluation: the judge is a
**different model** from the generator and is recorded in the results file, and
faithfulness is scored **per atomic claim** rather than per answer. RAGAS runs as an
independent cross-check when installed (`pip install -r requirements-eval.txt`).

Every failure is bucketed by cause, earliest cause winning — a generation error
downstream of a retrieval miss is not counted as a generation problem.

---

## Known limitations

- **Multi-hop full recall is 41.7%.** The largest single weakness. A query-decomposition
  or multi-step retrieval pass is the obvious next iteration, and the harness is
  already set up to measure whether it actually helps.
- **The eval set is small (45) and written by one person**, who also built the system.
  Confidence intervals on a 45-question set are wide; treat single-digit differences
  as noise.
- **No recital-answerable questions**, which overstates the recital down-weighting gain.
- **English only.** The Act is equally authentic in 24 languages; retrieval quality
  in the others is unmeasured.
- **`data/processed/chunks.jsonl` is generated**, committed only so the chunking can
  be inspected without running anything. CI rebuilds and re-tests it from source.
- Not legal advice.

---

## Repo layout

```
src/euactrag/    fetch → ingest (chunking) → index → retrieve → pipeline
eval/            qa_set.jsonl · metrics.py · judge.py · run_eval.py · report.py
app/             Streamlit UI showing answer beside its sources
tests/           corpus integrity, metric maths, citation parsing
deploy/          Hugging Face Space template
```

## Deploying the demo

```bash
make docker
```

`Dockerfile` builds the index at image-build time so a cold container answers its
first query without downloading a model. Built and verified on `linux/arm64`:
image is **2.8 GB**, ~4 min cold build, container reports healthy and answers
correctly from inside the image. The size is dominated by `torch` (635 MB) plus
transitive weight from `chromadb` (`onnxruntime`, `kubernetes`) and `streamlit`
(`pyarrow`) — swapping the encoder to ONNX Runtime, which `chromadb` already
installs, would remove `torch` entirely and is the obvious slimming step if the
image ever needs to be small.

For a free hosted demo, copy
[`deploy/HF_SPACE_README.md`](deploy/HF_SPACE_README.md) to the root of a Hugging
Face Space as `README.md`, push the repo, and set `GROQ_API_KEY` as a Space
**secret**.

---

## Attribution

Corpus: Regulation (EU) 2024/1689, Official Journal of the European Union, via the EU
Publications Office (CELEX 32024R1689). Reuse governed by Decision 2011/833/EU. Not
affiliated with or endorsed by the European Union.
