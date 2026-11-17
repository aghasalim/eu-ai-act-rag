"""Does retrieval hold up in French, German and Dutch?

    make eval-languages        # fetches the three texts if they are missing

The 33 answerable questions of the original set, translated
(eval/qa_translations.json), against the 2024 text of the Act in the same
language, article chunks, hybrid, k=6: the setting the published English
numbers were measured in. Every language, English included, uses the same
multilingual embedder, so a row differs from the English one by the language
and nothing else. The first row is the published English setup, for scale.

Each language is compared with English question by question: how many
questions lost or gained full recall, and the exact sign test on those.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.euactrag import config, retrieve  # noqa: E402
from src.euactrag.ingest import load_chunks  # noqa: E402
from eval.experiments import BASE, score, sign_test, variant  # noqa: E402

GRANULARITY = "article"
OUT = config.RESULTS_DIR / "languages.json"


def main() -> None:
    from eval.run_eval import load_qa

    english = load_qa("original")
    translated = json.loads((config.EVAL_DIR / "qa_translations.json").read_text())
    e5 = config.MULTILINGUAL_EMBED
    hybrid = variant(e5)
    dense = variant(e5, lambda q, k, v, g: retrieve.dense(q, k, v, g))
    bm25 = lambda q, k, v, g: retrieve.bm25(q, k, v, g)  # noqa: E731

    rows = {"English, bge-small (as published)":
            score(variant(BASE), english, config.TOP_K, config.ORIGINAL, GRANULARITY)}
    for lang in config.LANGUAGES:
        version = config.localized(config.ORIGINAL, lang)
        try:
            load_chunks(version, GRANULARITY)
        except FileNotFoundError:
            from src.euactrag import fetch, ingest
            for v in config.VERSIONS:
                fetch.fetch(config.localized(v, lang))
            ingest.main(lang)
        qa = english if lang == "en" else [
            {**q, "question": translated[lang][q["id"]]} for q in english if q["gold_units"]]
        r = rows[lang] = score(hybrid, qa, config.TOP_K, version, GRANULARITY)
        r["dense_full_recall"] = score(dense, qa, config.TOP_K, version, GRANULARITY)["full_recall"]
        r["bm25_full_recall"] = score(bm25, qa, config.TOP_K, version, GRANULARITY)["full_recall"]
        base = rows["en"]["per_question"]
        ids = sorted(base)
        up, down, p = sign_test([base[i] for i in ids], [r["per_question"][i] for i in ids])
        r["vs_english"] = {"better": up, "worse": down, "p": round(p, 4)}
        print(f"{lang}: hit {r['hit_rate']:.3f}  full {r['full_recall']:.3f}  "
              f"multi {r['multi_hop_full_recall']:.3f}  MRR {r['mrr']:.3f}  "
              f"dense {r['dense_full_recall']:.3f}  bm25 {r['bm25_full_recall']:.3f}  "
              f"+{up} -{down} p={p:.3f}", flush=True)
    OUT.write_text(json.dumps({"k": config.TOP_K, "embed": e5, "granularity": GRANULARITY,
                               "version": config.ORIGINAL, "rows": rows}, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
