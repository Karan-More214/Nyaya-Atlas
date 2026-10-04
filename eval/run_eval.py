"""Evaluate retrieval quality and abstention behaviour.

Usage:
    python -m eval.run_eval

Edit eval/eval_set.json to match the documents you indexed. Each item has:
  - question
  - expected_keywords: ANY of these (case-insensitive) appearing in a retrieved
    chunk counts as a hit
  - answerable: false for out-of-scope questions (system should abstain)

Metrics:
  - Hit@k      : share of answerable questions with a relevant chunk in the top-k
  - MRR        : mean reciprocal rank of the first relevant chunk
  - Abstention : share of out-of-scope questions correctly refused
"""
import json
from pathlib import Path

from src import config
from src.retriever import Retriever


def main():
    items = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
    retriever = Retriever()

    hits, rr_sum, n_answerable = 0, 0.0, 0
    abstain_ok, n_unanswerable = 0, 0

    for item in items:
        found = retriever.search(item["question"])
        if item["answerable"]:
            n_answerable += 1
            kws = [k.lower() for k in item["expected_keywords"]]
            rank = next(
                (
                    i
                    for i, r in enumerate(found["results"], start=1)
                    if any(k in r["text"].lower() for k in kws)
                ),
                None,
            )
            if rank:
                hits += 1
                rr_sum += 1.0 / rank
            print(f"[{'HIT ' if rank else 'MISS'}] rank={rank} | {item['question']}")
        else:
            n_unanswerable += 1
            refused = found["best_similarity"] < config.MIN_SIMILARITY
            abstain_ok += refused
            print(
                f"[{'OK  ' if refused else 'FAIL'}] best_sim={found['best_similarity']:.2f} "
                f"| {item['question']}"
            )

    print("\n=== Results ===")
    if n_answerable:
        print(f"Hit@{config.TOP_K}: {hits}/{n_answerable} = {hits / n_answerable:.0%}")
        print(f"MRR: {rr_sum / n_answerable:.3f}")
    if n_unanswerable:
        print(f"Abstention: {abstain_ok}/{n_unanswerable} = {abstain_ok / n_unanswerable:.0%}")


if __name__ == "__main__":
    main()
