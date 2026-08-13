# Evaluation results

Generated from `eval/results/eval_retrieval_only.json`. Regenerate with `make eval && make report`.

| setting | value |
|---|---|
| embed_model | `BAAI/bge-small-en-v1.5` |
| gen_mode | `hybrid` |
| top_k | `6` |
| rrf_k | `60` |
| n_chunks | `464` |
| n_questions | `45` |

## 1. Retrieval quality

Scored at *provision* level: retrieving any chunk of the correct article counts as a hit, since the user is directed to the right provision. `full_recall` requires **every** gold provision — the metric that matters for multi-hop questions.

### All strategies @ k=6

| strategy | hit rate | recall | full recall | precision | MRR | nDCG | s/query |
|---|---|---|---|---|---|---|---|
| **dense** | 81.8% | 67.2% | 51.5% | 19.6% | 0.521 | 0.537 | 0.402 |
| **bm25** | 84.9% | 73.2% | 63.6% | 18.4% | 0.561 | 0.571 | 0.002 |
| **hybrid** | 90.9% | 80.3% | 69.7% | 24.3% | 0.790 | 0.754 | 0.032 |

### Single-hop vs multi-hop

| strategy | type | hit rate | recall | full recall | MRR |
|---|---|---|---|---|---|
| dense | single_hop | 76.2% | 76.2% | 76.2% | 0.599 |
| dense | multi_hop | 91.7% | 51.4% | 8.3% | 0.385 |
| bm25 | single_hop | 81.0% | 81.0% | 81.0% | 0.537 |
| bm25 | multi_hop | 91.7% | 59.7% | 33.3% | 0.603 |
| hybrid | single_hop | 85.7% | 85.7% | 85.7% | 0.798 |
| hybrid | multi_hop | 100.0% | 70.8% | 41.7% | 0.778 |

### Sensitivity to k (3, 5, 6, 10)

| strategy | hit@3 | hit@5 | hit@6 | hit@10 | full@3 | full@5 | full@6 | full@10 |
|---|---|---|---|---|---|---|---|---|
| dense | 63.6% | 78.8% | 81.8% | 84.9% | 48.5% | 51.5% | 51.5% | 57.6% |
| bm25 | 60.6% | 81.8% | 84.9% | 93.9% | 45.5% | 63.6% | 63.6% | 75.8% |
| hybrid | 81.8% | 87.9% | 90.9% | 97.0% | 63.6% | 63.6% | 69.7% | 87.9% |

### Ablation: down-weighting recitals

Recitals restate the operative rules in flowing prose, so they match a natural-language question *better* than the terse article that actually contains the rule, and were crowding binding provisions out of the top-k. `w` is the weight a recital carries in rank fusion relative to a binding provision.

| w | hit rate | recall | full recall | MRR | nDCG |
|---|---|---|---|---|---|
| 1.0 | 87.9% | 76.3% | 66.7% | 0.581 | 0.600 |
| 0.75 | 90.9% | 79.3% | 69.7% | 0.790 | 0.748 |
| 0.5 ←default | 90.9% | 80.3% | 69.7% | 0.790 | 0.754 |
| 0.25 | 90.9% | 80.3% | 69.7% | 0.790 | 0.754 |
| 0.0 | 90.9% | 80.3% | 69.7% | 0.790 | 0.754 |

The gain is a **step, not a peak**: every `w < 1.0` scores the same, so the default is not an argmax fitted to this question set -- it is doing something structural, pushing non-binding text below binding text. `w=0.5` is kept rather than `w=0.0` because it scores identically while leaving recitals retrievable for interpretive questions.

> **Honest caveat.** Every gold label in this eval set is an article or annex, so an eval containing recital-answerable questions would show a smaller benefit. The measured gain is an upper bound.

## 2. Generation

_Not run: no LLM key was configured._
