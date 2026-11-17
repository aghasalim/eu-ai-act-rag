"""Fine-tune the embedding model on the synthetic (question, passage) pairs.

One run, fixed settings, no tuning against the eval questions. A tenth of the
chunks are held out with their questions and used only to print before/after
recall, so there is a number that the hand-written eval set had no part in.

    python train/finetune.py data/processed/chunks_2026-07-27_paragraph.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
BASE = "BAAI/bge-small-en-v1.5"
OUT = HERE / "bge-small-euaiact"
# Must match config.QUERY_PREFIX: the model is trained the way it is queried.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EPOCHS, BATCH, LR, SEED = 4, 64, 2e-5, 13


def eval_questions() -> set[str]:
    return {json.loads(l)["question"].strip().lower()
            for f in (HERE.parent / "eval").glob("qa_*.jsonl")
            for l in open(f, encoding="utf-8") if l.strip()}


def recall_at(model, pairs, chunks, k: int = 6) -> float:
    """Share of held-out questions whose own chunk is in the dense top k,
    searched against every chunk in the corpus."""
    ids = [c["chunk_id"] for c in chunks]
    docs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    qs = model.encode([QUERY_PREFIX + q for q, _ in pairs], normalize_embeddings=True)
    top = (-(qs @ docs.T)).argsort(axis=1)[:, :k]
    return sum(cid in {ids[j] for j in row}
               for (_, cid), row in zip(pairs, top)) / len(pairs)


def main(chunks_path: str) -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    text = {c["chunk_id"]: c["text"] for c in chunks}
    rows = [json.loads(l) for l in open(HERE / "synthetic_queries.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["questions"]]

    banned = eval_questions()
    leaked = [q for r in rows for q in r["questions"] if q.strip().lower() in banned]
    assert not leaked, f"synthetic questions that are also eval questions: {leaked}"

    random.Random(SEED).shuffle(rows)
    cut = len(rows) // 10
    held = [(q, r["chunk_id"]) for r in rows[:cut] for q in r["questions"]]
    train = [(q, r["chunk_id"]) for r in rows[cut:] for q in r["questions"]]
    print(f"{len(train)} training pairs, {len(held)} held out, "
          f"from {len(rows)} chunks")

    model = SentenceTransformer(BASE)
    before = recall_at(model, held, chunks)

    ds = Dataset.from_dict({"anchor": [QUERY_PREFIX + q for q, _ in train],
                            "positive": [text[cid] for _, cid in train]})
    SentenceTransformerTrainer(
        model=model, train_dataset=ds, loss=MultipleNegativesRankingLoss(model),
        args=SentenceTransformerTrainingArguments(
            output_dir=str(HERE / "runs"), num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH, learning_rate=LR, warmup_ratio=0.1,
            # Two questions about the same chunk in one batch would make the
            # chunk its own negative.
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            seed=SEED, save_strategy="no", report_to="none", logging_steps=10),
    ).train()

    after = recall_at(model, held, chunks)
    model.save(str(OUT))
    stats = {"base": BASE, "train_pairs": len(train), "held_out_pairs": len(held),
             "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seed": SEED,
             "held_out_recall_at_6": {"before": round(before, 4), "after": round(after, 4)}}
    (HERE / "finetune_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
