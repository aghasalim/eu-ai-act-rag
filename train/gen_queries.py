"""Write synthetic questions for every binding chunk, with a local model.

The embedding model is fine-tuned on (question, passage) pairs, and nobody has
written 2,000 questions by hand. A local instruction model writes three per
chunk. None of the hand-written eval questions is used here, and
train/finetune.py refuses to train if one turns up.

Runs on Apple silicon through mlx-lm. Appends as it goes, so a stopped run
resumes where it left off.

    python train/gen_queries.py data/processed/chunks_2026-07-27_paragraph.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"
OUT = Path(__file__).parent / "synthetic_queries.jsonl"
MIN_TOKENS = 30  # shorter chunks are headings and "deleted" stubs
BATCH = 16

PROMPT = """Passage from the EU AI Act ({citation}):

{body}

Write 3 different questions that this passage answers, as a JSON array of 3 strings:
1. a short question a compliance officer would type into a search box
2. a concrete scenario from a company that builds or uses an AI system
3. a question in everyday words that avoids the passage's legal terms

Rules: each question must be answerable from this passage alone. Do not mention \
article, annex, paragraph or point numbers. Do not copy more than four \
consecutive words from the passage. Output only the JSON array."""


def parse(reply: str) -> list[str]:
    m = re.search(r"\[.*\]", reply, re.S)
    try:
        qs = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in qs if isinstance(q, str) and q.strip().endswith("?")]


def main(chunks_path: str) -> None:
    from mlx_lm import batch_generate, load

    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    done = ({json.loads(l)["chunk_id"] for l in open(OUT, encoding="utf-8")}
            if OUT.exists() else set())
    todo = [c for c in chunks if c["kind"] != "recital"
            and c["n_tokens"] >= MIN_TOKENS and c["chunk_id"] not in done]
    if not todo:
        print(f"every chunk already has questions in {OUT.name}")
        return
    model, tok = load(MODEL)
    with open(OUT, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(
                    citation=c["citation"], body=c["body"])}],
                add_generation_prompt=True) for c in batch]
            replies = batch_generate(model, tok, prompts, max_tokens=300).texts
            for c, reply in zip(batch, replies):
                # A chunk with no usable questions is still recorded, so a
                # resume does not spend another generation on it.
                f.write(json.dumps({"chunk_id": c["chunk_id"], "questions": parse(reply)},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(i + BATCH, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
