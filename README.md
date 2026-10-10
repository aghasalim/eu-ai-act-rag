# EU AI Act RAG, with an evaluation I actually ran

**[▶ Live demo](https://eu-ai-act-rag-eval.streamlit.app/)** · every
answer shows the passages it came from.

[![ci](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/ci.yml)
[![demo-link](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/demo.yml/badge.svg)](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/demo.yml)
[![publish-image](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/publish-image.yml/badge.svg)](https://github.com/aghasalim/eu-ai-act-rag/actions/workflows/publish-image.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23003624.svg)](https://doi.org/10.5281/zenodo.23003624)

I built a question-answering system over Regulation (EU) 2024/1689 (the EU AI Act),
as amended by Regulation (EU) 2026/1744 (the "Digital Omnibus on AI"). The corpus is
versioned by the date of the text. `2026-07-27` is the consolidated text and the default.
`2024-07-12` is the original Official Journal text, which I kept so the two can be compared.

Most RAG projects, mine included, stop at "look, it answers questions" and never check
whether the answer is in the documents retrieved. I wanted to measure that properly.
So I wrote 45 questions by hand, scored the system against them and wrote down the
failures. Retrieval is decent. 90.9% of questions get at least one
correct provision. It's bad at questions that need two articles at once, though, and
only 41.7% of those get everything they need. I'd rather show you
that number than hide it. Every figure here is recomputed from the per-question
records by independent implementations in `verify/`, and CI fails if any of them
disagree.

Long write-up: **[notes/METHODS.md](notes/METHODS.md)**. Per-question numbers:
**[RESULTS.md](RESULTS.md)**.

---

## Retrieval

![retrieval strategies across k](eval/figures/retrieval-across-k.png)

45 hand-written questions over 464 chunks. BM25 beats dense on full recall at every k,
and fusing the two beats both. That last gap is only two questions out of 33, though.
It doesn't survive a paired test, which I only found out by
[recomputing it](#every-number-here-is-recomputed-in-another-language). Retrieval
scoring needs no LLM, so these numbers are free to reproduce.

### Retrieval, k=6

<!-- RETRIEVAL_TABLE:START -->
| strategy | hit rate | recall | full recall | MRR | nDCG | s/query |
|---|---|---|---|---|---|---|
| dense (bge-small) | 81.8% | 67.2% | 51.5% | 0.521 | 0.537 | 0.369 |
| BM25 | 84.9% | 73.2% | 63.6% | 0.561 | 0.571 | 0.002 |
| **hybrid (RRF)** | **90.9%** | **80.3%** | **69.7%** | **0.795** | **0.756** | 0.019 |
<!-- RETRIEVAL_TABLE:END -->

"Hit rate" means at least one required article showed up. "Full recall" means *all* of
them did, and that is the column I care about.

![single-hop against multi-hop](eval/figures/single-vs-multi-hop.png)

Multi-hop questions are where full recall drops. Hybrid finds something relevant
for 100% of them and everything they need for 41.7%. Keyword search beating embeddings
was the other surprise. I break both down in
[notes/METHODS.md](notes/METHODS.md#three-things-i-did-not-expect).

![per-question coverage as k grows](eval/figures/coverage-across-k.gif)

*Each tile is one of the 33 answerable questions, coloured by how much of what
it needs hybrid retrieval has found. Only k changes across the frames, 3 then
5 then 6 then 10, and the tiles that never fill in are the multi-hop ones.*

![down-weighting recitals is a step, not a peak](eval/figures/recital-ablation.png)

Recitals are the non-binding "whereas" paragraphs. They restate the rules in flowing
prose, and they were crowding binding articles out of the top-k. Down-weighting them moved MRR from
0.601 to 0.795. Any weight below 1.0 gives nearly the same result. MRR and full recall
are identical across them, and nDCG only creeps from 0.751 to 0.756. So the gain is a
step. It isn't a peak I fitted to 45 questions.

## Answers

Answers come from `openai/gpt-oss-20b` and are graded by `qwen/qwen3.6-27b`. That's a
different model family, so the grader isn't marking its own work. This covers all 45 questions.

| metric | value |
|---|---|
| **Faithfulness** (claims entailed by retrieved text, averaged over the 26 questions it answered that have an answer) | **90.2%** |
| **Citation validity** (citations pointing at retrieved passages) | **100%** |
| **Correct abstention** on out-of-scope questions | **100%** (12/12) |
| **Hallucination rate** on out-of-scope questions | **0%** |
| Answer accuracy, strict | 66.7% |
| Answer accuracy, incl. partially correct | 69.7% |
| False abstention (refused a question it could answer) | 21.2% |

![answer quality and the refusal trade](eval/figures/answer-quality.png)

It refused 12 out of 12 out-of-scope questions and hallucinated on none, even the ones
I wrote to bait it. It leans toward refusing, which is why false abstention is 21%. The
per-type breakdown is in [notes/METHODS.md](notes/METHODS.md#answer-quality-broken-out),
along with the `LLM_MAX_TOKENS` bug that was costing 12 points of accuracy.

![where the 45 questions end up](eval/figures/failure-modes.png)

Twelve outcomes aren't ok. Six of those are retrieval failures, three complete misses
and three partial. Four more are refusals of answerable questions. Only two are
generation faults where the right evidence was there, so retrieval is where I'd spend the effort.

## The law changed under it

I built and measured this on the 2024 text. On 27 July 2026 the Omnibus changed it.
It moved the high-risk dates, Annex III to 2 December 2027 and Annex I to 2 August 2028.
It added two Article 5 prohibitions. It also replaced Article 4 and added Articles 4a
and 60a, among other changes. So I wrote 11 more questions that only the amended text
can answer. Each one comes with the exact wording its answer rests on, and I ran
retrieval on both texts. The changes I checked and the quote behind every question are in
[notes/AMENDMENTS.md](notes/AMENDMENTS.md).

Hybrid retrieval, k=6:

<!-- STALE_TABLE:START -->
| questions | text searched | answerable | hit rate | full recall | MRR | current wording in top k |
|---|---|---|---|---|---|---|
| original | 2024-07-12 | 33 | 90.9% | 69.7% | 0.795 | n/a |
| original | 2026-07-27 | 33 | 90.9% | 69.7% | 0.785 | n/a |
| amended | 2024-07-12 | 11 | 63.6% | 45.5% | 0.636 | 0.0% |
| amended | 2026-07-27 | 11 | 81.8% | 63.6% | 0.773 | 63.6% |

On the amended questions the 2024 text still finds a provision with the right number for 63.6% of them, and the current wording for 0.0%. On the 2026 text that is 81.8% and 63.6%.
<!-- STALE_TABLE:END -->

A stale RAG system hides this kind of failure. If you measure it the usual way, by
whether the right article number comes back, the old corpus still looks respectable on
questions about the new law. That's because most amendments rewrite an article without
renumbering it. But not one retrieved chunk holds the current wording. So any answer
built on it states the old rule with a correct-looking citation. There are two articles
it can't return at all, 4a and 60a, because they didn't exist in 2024.

The two plain date questions (a01, a02) miss Article 113 on both texts. Their top 6 is
all articles about high-risk systems (6, 8, 43 and so on), and none of them is the
article that holds the dates. It's the same weakness that cost s02 on the original run.
Also, the answer numbers above were measured on the 2024 text. I haven't re-run
generation on the amended one, so there's no answer-level comparison yet, only retrieval.

Three of the original 45 questions are now out of date (m04 superseded, s15 and m12
incomplete). They're marked in `qa_set.jsonl` and left as they were, since they're
right for the text the published numbers were measured on.

## Citing the paragraph, not the article

People cite a legal answer as "Article 6(3)" rather than "Article 6", so I added a
second chunking. It makes one chunk per numbered paragraph, or per top-level point where an
article has no numbered paragraphs (Article 3's definitions, Article 113's dates).
Every line is labelled with its place in the outline, so a chunk knows it's
Article 5(1) and holds points (f) to (h). The text is the same and only the cut differs.
The 2024 text gives 921 chunks instead of 464. The 2026 text gives 994 instead of 498.

Each of the 56 questions with an answer now also has gold paragraph references
(`gold_refs`, e.g. `art_5(1)(f)`). A test checks that every one of them exists
in the corpus. The new metrics treat each retrieved chunk as a citation of whatever it's
labelled as, and they compare at paragraph level. So Article 5(1)(f) and 5(1)(h) count as
the same paragraph, while 6(3) and 6(4) don't. Citation precision is the share of
distinct cited paragraphs that are gold.

Hybrid retrieval, k=6 (`make eval-paragraph`):

| questions | chunks | hit rate | full recall | MRR | paragraph hit | paragraph full recall | paragraph MRR | citation precision |
|---|---|---|---|---|---|---|---|---|
| original, 2024 text | article | 90.9% | 69.7% | 0.790 | 0.0% | 0.0% | 0.000 | 0.000 |
| original, 2024 text | paragraph | 97.0% | 75.8% | 0.843 | 90.9% | 75.8% | 0.757 | 0.176 |
| amended, 2026 text | article | 81.8% | 63.6% | 0.773 | 0.0% | 0.0% | 0.000 | 0.000 |
| amended, 2026 text | paragraph | 81.8% | 63.6% | 0.818 | 81.8% | 54.5% | 0.773 | 0.155 |

The article rows are zero by construction. An article chunk can only be cited as
the whole article. Only one gold reference is article level (s02, the general application
date in Article 113's unnumbered text), and hybrid misses it.
Citation precision at k=6 also has a low ceiling. Most questions have one or two gold
paragraphs and six citations. A perfect ranker would score 0.232 on the original
set and 0.212 on the amended one. At k=3 it's 0.308 against a ceiling of 0.465.

I didn't expect smaller chunks to help at article level too, but they did on the
original questions. Full recall went from 69.7% to 75.8%, which is two more questions. On 33
questions that isn't a significant gap. It didn't cost anything either, so
search and the app now use paragraph chunks. The published answer numbers above
were measured on article chunks, and `make eval` still reproduces them that way.

## A bigger question set

45 questions was too few to say much, so I wrote 67 more in
`eval/qa_extended.jsonl` from the 2026 text. That makes 123 written, and 122 in use once the
superseded `m04` is dropped. Each one has gold
article and paragraph references, a short gold answer and the exact wording the
answer rests on. `make check-qa` (also a test in CI) checks all three files against
the corpus. Every gold reference has to exist and every quote has to sit inside a gold
paragraph. In the new set, every gold paragraph also has to hold a quote. The check
works out which questions quote wording the 2024 text doesn't have, too. 13 of the 67 do.

| topic | questions |
|---|---|
| high-risk classification and obligations | 28 |
| prohibited practices | 7 |
| general-purpose AI models | 7 |
| transparency | 5 |
| penalties | 6 |
| dates and transitional periods | 6 |
| provisions added by the 2026 amendments | 8 |

Hybrid retrieval on the 2026 text, k=6 (`make eval-extended`). "All" is the three
sets together, minus m04, which the amendments superseded. That leaves 110 questions with an answer.

| questions | chunks | answerable | hit rate | full recall | MRR | paragraph full recall | citation precision |
|---|---|---|---|---|---|---|---|
| new 67 | article | 67 | 95.5% | 89.6% | 0.812 | 0.0% | 0.000 |
| new 67 | paragraph | 67 | 98.5% | 92.5% | 0.865 | 86.6% | 0.171 |
| all | article | 110 | 92.7% | 80.9% | 0.799 | 0.9% | 0.002 |
| all | paragraph | 110 | 96.4% | 83.6% | 0.853 | 79.1% | 0.169 |

The new questions are easier than the original 45, and I think I know why. I wrote
them with the paragraph open, so they reuse its words, and that's what BM25 rewards.
The original set was written to be harder and still is. I'd treat the new numbers as a
regression check for the paragraph chunker and the amended text. They don't tell you
how the system does on real users' questions. Paragraph chunks were
better than article chunks on all of it, by 2.7 points of full recall on the 110.
There are 13 questions that need the amended wording. With paragraph chunks, 10 of them
get it in the top 6. With article chunks, 11 do.

## Checking the judge

The answer grades above come from an LLM judge, so I graded 41 of the same answers
by hand. I worked from the text of the Act and didn't look at the judge's grade. They're
every non-empty answer that didn't abstain in two runs of the same generator, with
answers identical across the runs counted once. The labels, with a note on each
mistake, are in `eval/human_labels.jsonl`, and I made all of them myself. `make agreement`
compares them with the stored outputs, so it needs no API key.

| what is compared | items | agreement | Cohen's kappa |
|---|---|---|---|
| judge grade, correct / partial / incorrect | 41 | 82.9% | 0.52 |
| judge grade, correct against not correct | 41 | 82.9% | 0.48 |
| abstention rule (`NOT_IN_CORPUS` in the answer) | 45 | 100% | 1.00 |

All 7 disagreements are about "partial". The judge marked 5 answers partial that I
marked correct. Four of those were for leaving out a detail of the reference that I didn't
think the question needed (s13, s21, m01, m10). One was for a sentence cut off after the
answer was already given (s17). It also marked
m04 correct twice where I marked it partial. The list of articles is right, but the
answer describes Articles 102 to 109 and 112 wrongly, and the judge only compares
against the reference. So it's strict about omissions but doesn't catch wrong extra
claims. On the published run the disagreements cancel out. Strict accuracy is 66.7% and
lenient is 69.7% under either set of grades. Every answer either of us called incorrect,
both of us did.

The citation validity number depends on the citation parser. It agrees with my
reading of what each answer cites on 80.5% of answers. It never invents a citation,
so precision is 100%. It does miss 20.4% of them (recall 79.6%). Most misses are citations
outside brackets, like "Article 55(c)" in bold or "Article 49(2) requires" in running text.
Two are brackets left unclosed by an answer that was cut off. So the citation validity
figure is computed over the bracketed citations only.

I haven't checked the claim-level faithfulness judge. That needs each claim read
against the retrieved passages, and I haven't done it yet. There's no API key
on this machine, so the judge hasn't been re-run either. `eval/agreement.py --rejudge
MODEL` grades the same 41 answers with a live judge and adds it to the comparison
once a key is in `.env`.

## Method, briefly

I take the official XHTML from the EU Publications Office Cellar API (CELEX `32024R1689`)
and chunk it on the document's own articles, recitals and annexes. The answer to a
legal question is a citation, so that's the natural cut. That gives 113 articles + 180 recitals + 13 annexes → **464 chunks**,
all under the encoder's 512-token limit. The amended text (CELEX `02024R1689-20260727`)
comes from the same API in different markup. It has no preamble, so the unamended
recitals are carried over: 119 articles + 180 recitals + 14 annexes → 498 chunks.
Retrieval is `BAAI/bge-small-en-v1.5` in Chroma fused with BM25 by Reciprocal Rank Fusion.
RRF ranks by position, so there's no scaling constant to fit. There's more detail in
[notes/METHODS.md](notes/METHODS.md#3-method), and the test set is described in
[notes/METHODS.md](notes/METHODS.md#4-the-test-set).

## Running it

```bash
make setup && make corpus && make index
make eval-retrieval
```

That reproduces every retrieval number above. It needs no API key and makes no LLM calls, so it's free.
`make eval-versions` adds the runs on the amended text and the amended questions.
`make eval-paragraph` compares article and paragraph chunks, and `make eval-extended`
runs the 122-question set.

For generated answers, add a free [Groq](https://console.groq.com/keys) key:

```bash
cp .env.example .env && echo "GROQ_API_KEY=your_key_here" >> .env
make eval && make report && make app
```

### As a package

The code installs as `euactrag`, with a command and two functions:

```bash
pip install .            # or: pip install dist/euactrag-0.1.0-py3-none-any.whl
euactrag fetch && euactrag ingest && euactrag index
euactrag search "Is emotion recognition at work prohibited?" -k 3
euactrag ask "What fine applies to a prohibited practice?"    # needs an LLM key
```

```python
import euactrag
hits = euactrag.search("When do the Annex III rules apply?", k=3)
print(hits[0]["citation"], hits[0]["ref"])
```

Outside a checkout the corpus and index go to `~/.cache/euactrag`, or wherever
`EUACTRAG_DATA` points. `make dist` builds the sdist and wheel and checks them, and
CI does the same on every push and installs the wheel on its own. It isn't on PyPI.

Or pull the container:

```bash
docker run -p 8501:8501 ghcr.io/aghasalim/eu-ai-act-rag:latest
```

`make docker` builds it locally. Image size and hosting notes are in
[notes/METHODS.md](notes/METHODS.md#7-docker).

## Limitations

- **41.7% full recall on multi-hop questions.** It's the biggest weakness by far.
- **I wrote every question myself** for a system I also built. The headline numbers
  rest on the original 45, and small differences there are noise. The 67 added later
  share wording with the text they were written from, so they flatter retrieval.
- **The answer numbers are for the 2024 text.** Retrieval has been measured on both
  texts; generation only on the original.
- **The consolidated text has no legal effect.** Only the Official Journal texts are
  authentic. I checked every change I rely on against the amending act itself.
- **No questions where a recital is the right answer.** That makes the recital
  down-weighting look better than it probably is.
- **English only.** The Act is equally valid in 24 languages and I've tested one.
- This is a student project and it isn't legal advice. Please don't make compliance decisions with it.

Full list in [notes/METHODS.md](notes/METHODS.md#5-limitations).

## Repository layout

```
src/euactrag/    fetch → ingest (chunking) → index → retrieve → pipeline
eval/            qa_set.jsonl · qa_amended.jsonl · metrics.py · judge.py · run_eval.py · report.py
app/             Streamlit UI, shows the answer next to its sources
tests/           corpus integrity, metric maths, citation parsing
deploy/          Hugging Face Space template
notes/           METHODS.md, the long-form write-up · AMENDMENTS.md, the 2026 changes
verify/          the same numbers, recomputed independently
```

## Credit

The corpus is Regulation (EU) 2024/1689 from the Official Journal of the European Union,
via the EU Publications Office (CELEX 32024R1689), and its consolidated text as amended
by Regulation (EU) 2026/1744 (CELEX 02024R1689-20260727). Reuse is covered by Decision
2011/833/EU. Nothing here is affiliated with or endorsed by the EU.

The code is MIT ([LICENSE](LICENSE)). The corpus is not mine to licence, so its
attribution lives in [NOTICE](NOTICE).
