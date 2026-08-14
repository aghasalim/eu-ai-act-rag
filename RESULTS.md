# Evaluation results

Generated from `eval/results/eval_latest.json`. Regenerate with `make eval && make report`.

| setting | value |
|---|---|
| embed_model | `BAAI/bge-small-en-v1.5` |
| gen_model | `openai/gpt-oss-20b` |
| judge_model | `qwen/qwen3.6-27b` |
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
| **dense** | 81.8% | 67.2% | 51.5% | 19.6% | 0.521 | 0.537 | 0.269 |
| **bm25** | 84.9% | 73.2% | 63.6% | 18.4% | 0.561 | 0.571 | 0.001 |
| **hybrid** | 90.9% | 80.3% | 69.7% | 24.3% | 0.795 | 0.756 | 0.010 |

### Single-hop vs multi-hop

| strategy | type | hit rate | recall | full recall | MRR |
|---|---|---|---|---|---|
| dense | single_hop | 76.2% | 76.2% | 76.2% | 0.599 |
| dense | multi_hop | 91.7% | 51.4% | 8.3% | 0.385 |
| bm25 | single_hop | 81.0% | 81.0% | 81.0% | 0.537 |
| bm25 | multi_hop | 91.7% | 59.7% | 33.3% | 0.603 |
| hybrid | single_hop | 85.7% | 85.7% | 85.7% | 0.798 |
| hybrid | multi_hop | 100.0% | 70.8% | 41.7% | 0.792 |

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
| 1.0 | 87.9% | 76.3% | 66.7% | 0.586 | 0.602 |
| 0.75 | 90.9% | 79.3% | 69.7% | 0.795 | 0.750 |
| 0.5 ←default | 90.9% | 80.3% | 69.7% | 0.795 | 0.756 |
| 0.25 | 90.9% | 80.3% | 69.7% | 0.795 | 0.756 |
| 0.0 | 90.9% | 80.3% | 69.7% | 0.795 | 0.756 |

The gain is a **step, not a peak**: every `w < 1.0` scores the same, so the default is not an argmax fitted to this question set -- it is doing something structural, pushing non-binding text below binding text. `w=0.5` is kept rather than `w=0.0` because it scores identically while leaving recitals retrievable for interpretive questions.

> **Honest caveat.** Every gold label in this eval set is an article or annex, so an eval containing recital-answerable questions would show a smaller benefit. The measured gain is an upper bound.

## 2. Answer quality, faithfulness and abstention

> **Coverage: 41 of 45 questions.** Groq's free tier meters tokens *per day*, and a full run exceeds that allowance. Missing: `m10`, `m11`, `m12`, `m13`. The harness checkpoints every row, so rerunning `make eval` after the quota resets completes the set without repeating work. All 12 out-of-scope questions and all 21 single-hop questions were scored.

| metric | value | what it means |
|---|---|---|
| Answer accuracy (strict) | 58.6% | judged fully equivalent to the hand-written reference |
| Answer accuracy (incl. partial) | 69.0% | correct but possibly missing an element |
| **Faithfulness** | 92.6% | share of atomic claims entailed by the retrieved passages |
| **Citation validity** | 100.0% | share of inline citations pointing at a passage actually retrieved |
| **Correct abstention** | 100.0% | out-of-scope questions correctly refused |
| Hallucination rate (out-of-scope) | 0.0% | out-of-scope questions answered anyway |
| False abstention | 17.2% | answerable questions wrongly refused |
| Mean latency | 43.80s | _not a real latency figure_ — the client sleeps to stay under the free tier's token-per-minute cap, so this is dominated by throttling, not by the model |

### By question type

| type | n | accuracy (strict) | faithfulness |
|---|---|---|---|
| single_hop | 21 | 71.4% | 96.4% |
| multi_hop | 8 | 25.0% | 75.4% |
| unanswerable | 12 | 100.0% | — |

## 3. Where it fails

| failure mode | n | meaning |
|---|---|---|
| `ok` | 29 | Answered correctly (or correctly refused). |
| `retrieval_miss` | 3 | No gold provision in the top-k. The generator never had a chance. |
| `incomplete_answer` | 3 | Right direction, missing a required element (e.g. an exception). |
| `generation_error` | 2 | Right passages retrieved, wrong answer produced. |
| `partial_retrieval` | 2 | Multi-hop question where only some required provisions were retrieved. |
| `false_abstention` | 2 | Refused although the evidence was retrieved. |

### Every failing question

| id | type | failure | question | why |
|---|---|---|---|---|
| `s01` | single_hop | `generation_error` | What is the maximum administrative fine for non-compliance with the prohibited AI practice | The candidate provides the wrong fine amount (EUR 1,500,000) instead of the correct maximum of EUR 35,000,000 or 7% of turnover. |
| `s02` | single_hop | `retrieval_miss` | From what date does the EU AI Act generally apply? |  |
| `s12` | single_hop | `retrieval_miss` | Is testing in real world conditions covered by the exclusion for research, testing and dev |  |
| `s13` | single_hop | `incomplete_answer` | What must providers of general-purpose AI models do in relation to copyright? | The candidate answer correctly identifies the main obligations but omits the reference to using state-of-the-art technologies to comply with these req |
| `s17` | single_hop | `incomplete_answer` | For an SME, is the administrative fine capped at the percentage or at the fixed amount? | The candidate answer is incomplete because the final sentence is cut off and does not finish the explanation. |
| `s20` | single_hop | `retrieval_miss` | Which penalty tier applies to a deployer that breaches its obligations under Article 26? |  |
| `m01` | multi_hop | `incomplete_answer` | A hospital deploys an emotion recognition system on patients for medical reasons. Is that  | The candidate correctly identifies that the system is not prohibited and mentions the transparency obligation, but omits the reference's requirement t |
| `m02` | multi_hop | `generation_error` | An AI system listed in Annex III performs profiling of natural persons. Can its provider r | The candidate answer is empty and fails to provide the required response. |
| `m03` | multi_hop | `partial_retrieval` | A general-purpose AI model was trained with 10^26 floating point operations and is release | The candidate incorrectly concludes that the exemption applies, failing to recognize that a model trained with 10^26 FLOPs is presumed to have systemi |
| `m05` | multi_hop | `false_abstention` | Real-time remote biometric identification in public spaces for law enforcement is restrict |  |
| `m06` | multi_hop | `partial_retrieval` | An AI system is used to filter job applications. Is it high-risk, and what fine would its  | The candidate answer is empty and fails to address the question. |
| `m07` | multi_hop | `false_abstention` | When do the general-purpose AI obligations in Chapter V start to apply, and by when must m |  |

### Claims the judge could not trace to a retrieved passage

| question | claim # | judge reason |
|---|---|---|
| `s05` | 3 | Article 2(3) excludes AI systems not placed on the market/put into service in the Union only 'where the output is used i |
| `s08` | 0 | The excerpts state that reports must be made immediately only in specific cases (widespread infringement, death, or seri |
| `s17` | 1 | Article 99 generally sets fines as 'whichever is higher' (paras 3, 4, 5). The 'lower' cap applies only to SMEs under par |
| `m01` | 2 | The excerpts define regulations and prohibitions but do not contain any factual information about a specific hospital or |
| `m01` | 3 | The excerpts do not identify a specific 'system' to which the obligation applies; they only state the general rule that  |
| `m04` | 2 | The excerpts do not define the content of Articles 102 to 109; they only list them as applicable provisions in Article 2 |
| `m04` | 3 | The excerpts do not define the content of Article 112; they only list it as an applicable provision in Article 2(2). Cla |
| `m08` | 0 | Article 5(f) states that the prohibition on inferring emotions in education institutions has an exception: 'except where |
