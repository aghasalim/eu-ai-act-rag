# Evaluation results

Generated from `eval/results/eval_latest.json`. Regenerate with `make eval && make report`.

| setting | value |
|---|---|
| embed_model |`BAAI/bge-small-en-v1.5` |
| gen_model |`openai/gpt-oss-20b` |
| judge_model |`qwen/qwen3.6-27b` |
| gen_mode |`hybrid` |
| top_k |`6` |
| rrf_k |`60` |
| n_chunks |`464` |
| n_questions |`45` |

## 1. Retrieval quality

Scored at *provision* level: retrieving any chunk of the correct article counts as a hit, since the user is directed to the right provision. `full_recall` requires **every** gold provision, the metric that matters for multi-hop questions.

### All strategies @ k=6

| strategy | hit rate | recall | full recall | precision | MRR | nDCG | s/query |
|---|---|---|---|---|---|---|---|
| **dense** | 81.8% | 67.2% | 51.5% | 19.6% | 0.521 | 0.537 | 0.369 |
| **bm25** | 84.9% | 73.2% | 63.6% | 18.4% | 0.561 | 0.571 | 0.002 |
| **hybrid** | 90.9% | 80.3% | 69.7% | 24.3% | 0.790 | 0.754 | 0.019 |

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

The gain is a **step, not a peak**: nearly all of it comes from dropping below `w=1.0`, and the weights under that barely differ, nDCG only 0.748 to 0.754. So the default is not an argmax fitted to this question set, it is doing something structural, pushing non-binding text below binding text. `w=0.5` is kept rather than `w=0.0` because it ties on every column above while leaving recitals retrievable for interpretive questions.

> **Honest caveat.** Every gold label in this eval set is an article or annex, so an eval containing recital-answerable questions would show a smaller benefit. The measured gain is an upper bound.

## 2. Answer quality, faithfulness and abstention

| metric | value | what it means |
|---|---|---|
| Answer accuracy (strict) | 66.7% | judged fully equivalent to the hand-written reference |
| Answer accuracy (incl. partial) | 69.7% | correct but possibly missing an element |
| **Faithfulness** | 90.2% | share of atomic claims entailed by the retrieved passages |
| **Citation validity** | 100.0% | share of inline citations pointing at a passage actually retrieved |
| **Correct abstention** | 100.0% | out-of-scope questions correctly refused |
| Hallucination rate (out-of-scope) | 0.0% | out-of-scope questions answered anyway |
| False abstention | 21.2% | answerable questions wrongly refused |
| Mean latency | 54.76s | _not a real latency figure_, the client sleeps to stay under the free tier's token-per-minute cap, so this is dominated by throttling, not by the model |

### By question type

| type | n | accuracy (strict) | faithfulness |
|---|---|---|---|
| single_hop | 21 | 76.2% | 97.2% |
| multi_hop | 12 | 50.0% | 74.3% |
| unanswerable | 12 | 100.0% | - |

_RAGAS cross-check unavailable: ragas not installed (ModuleNotFoundError)_

## 3. Where it fails

| failure mode | n | meaning |
|---|---|---|
|`ok` | 33 | Answered correctly (or correctly refused). |
|`false_abstention` | 4 | Refused although the evidence was retrieved. |
|`retrieval_miss` | 3 | No gold provision in the top-k. The generator never had a chance. |
|`partial_retrieval` | 3 | Multi-hop question where only some required provisions were retrieved. |
|`generation_error` | 1 | Right passages retrieved, wrong answer produced. |
|`incomplete_answer` | 1 | Right direction, missing a required element (e.g. an exception). |

### Every failing question

| id | type | failure | question | why |
|---|---|---|---|---|
|`s01` | single_hop |`generation_error` | What is the maximum administrative fine for non-compliance with the prohibited AI practice | The candidate provides the wrong fine amount (EUR 1,500,000 instead of EUR 35,000,000 or 7% of turnover) and cites the incorrect article. |
|`s02` | single_hop |`retrieval_miss` | From what date does the EU AI Act generally apply? |  |
|`s12` | single_hop |`retrieval_miss` | Is testing in real world conditions covered by the exclusion for research, testing and dev |  |
|`s20` | single_hop |`retrieval_miss` | Which penalty tier applies to a deployer that breaches its obligations under Article 26? |  |
|`m03` | multi_hop |`partial_retrieval` | A general-purpose AI model was trained with 10^26 floating point operations and is release | The candidate incorrectly concludes that the exemption applies, failing to recognize that 10^26 FLOPs exceeds the 10^25 threshold for systemic risk, w |
|`m05` | multi_hop |`false_abstention` | Real-time remote biometric identification in public spaces for law enforcement is restrict |  |
|`m06` | multi_hop |`false_abstention` | An AI system is used to filter job applications. Is it high-risk, and what fine would its  |  |
|`m07` | multi_hop |`partial_retrieval` | When do the general-purpose AI obligations in Chapter V start to apply, and by when must m | The candidate incorrectly states that the obligations start applying when the Regulation enters into force, whereas the reference specifies they apply |
|`s21` | single_hop |`incomplete_answer` | A provider concludes that its Annex III AI system is not high-risk. What must it do before | The candidate answer correctly identifies the documentation and registration requirements but omits the obligation to provide the documentation to nat |
|`m11` | multi_hop |`false_abstention` | What must a chatbot provider disclose to users, and what fine applies if it fails to? |  |
|`m12` | multi_hop |`false_abstention` | The prohibitions in Article 5 became applicable before the rest of the Regulation. From wh |  |
|`m13` | multi_hop |`partial_retrieval` | A provider established in the United States places a general-purpose AI model on the Union | The candidate answer correctly confirms the Regulation's applicability to the US provider and accurately lists the required documentation, including t |

### Claims the judge could not trace to a retrieved passage

| question | claim # | judge reason |
|---|---|---|
|`s05` | 3 | Article 2(3) excludes AI systems not placed on the market/put into service in the Union only 'where the output is used i |
|`s08` | 0 | The excerpts state that reports must be made immediately only in specific cases (widespread infringement, death, or seri |
|`m01` | 0 | The excerpts do not specify the context or type of system being deployed, so it is impossible to determine from the text |
|`m01` | 3 | The excerpts do not mention 'patients' or a medical context for the general transparency obligation in Article 50(3); th |
|`m01` | 5 | The excerpts do not identify a specific 'system in question' or its purpose, so it is impossible to determine from the t |
|`m02` | 2 | The excerpts do not contain an 'Article 6(1)(d)'. The 'notwithstanding' clause is in Article 6(3), second subparagraph,  |
|`m02` | 3 | The excerpts do not contain an 'Article 6(1)(d)'. The rule regarding profiling is in Article 6(3), second subparagraph:  |
|`m02` | 4 | The excerpts define general rules and categories but do not describe a specific 'system in question' or state that any s |
|`m02` | 5 | The excerpts do not identify a specific system to classify. While they state profiling systems are high-risk, they do no |
|`m02` | 6 | The excerpts do not identify a specific system or provider. While they state profiling systems are high-risk (thus ineli |
|`m03` | 3 | The excerpts do not mention the specific value '10^26 floating-point operations' nor do they state that training size is |
|`m04` | 2 | The excerpts do not define the content of Articles 102 to 109; they only list them as applicable provisions in Article 2 |
|`m04` | 3 | The excerpts do not define the content of Article 112; they only list it as an applicable provision in Article 2(2). Cla |
|`m07` | 2 | The excerpts do not state when the Regulation enters into force or that obligations apply from that date; they only spec |
|`m07` | 3 | The excerpts do not discuss the general start date of the Regulation's application or assert that Article 53 does not sp |
|`m13` | 18 | The requirement is set out in Article 54(3)(b), not Article 54(b)(ii). There is no 'Article 54(b)(ii)' in the provided t |
