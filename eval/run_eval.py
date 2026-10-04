"""Evaluate retrieval quality and abstention behaviour.

Usage:
    python -m eval.run_eval

Edit eval/eval_set.json to match the documents you indexed. Each item has:
  - question
  - expected_keywords: ANY of these (case-insensitive, whitespace-normalised)
    appearing in a retrieved chunk counts as a hit
  - expected_source (optional): the chunk must also come from this file
  - answerable: false for out-of-scope questions (system should abstain)

Metrics:
  - Hit@k             : share of answerable questions with a relevant chunk in the top-k
  - MRR               : mean reciprocal rank of the first relevant chunk
  - Abstention        : share of out-of-scope questions correctly refused
  - Answered          : share of answerable questions NOT refused (guards against a
                        threshold so high that everything is refused)
"""
import json
import re
from pathlib import Path

from src import config
from src.retriever import Retriever

EVAL_SET = Path(__file__).parent / "eval_set.json"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def load_items() -> list[dict]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))


def first_hit_rank(item: dict, results: list[dict]) -> int | None:
    kws = [_norm(k) for k in item["expected_keywords"]]
    src = item.get("expected_source")
    for i, r in enumerate(results, start=1):
        if src and r["source"] != src:
            continue
        if any(k in _norm(r["text"]) for k in kws):
            return i
    return None


def collect(retriever: Retriever, items: list[dict], k: int) -> list[dict]:
    """Run retrieval once per question (k results) and keep what scoring needs."""
    return [
        {"item": it, **retriever.search(it["question"], k=k)} for it in items
    ]


def score(runs: list[dict], k: int, min_sim: float) -> dict:
    """Score collected runs at cut-off k and abstention threshold min_sim."""
    hits, rr, n_ans, answered, abstain_ok, n_out = 0, 0.0, 0, 0, 0, 0
    for run in runs:
        refused = run["best_similarity"] < min_sim
        if run["item"]["answerable"]:
            n_ans += 1
            answered += not refused
            rank = first_hit_rank(run["item"], run["results"][:k])
            if rank:
                hits += 1
                rr += 1.0 / rank
        else:
            n_out += 1
            abstain_ok += refused
    return {
        "hit": hits / n_ans if n_ans else 0.0,
        "mrr": rr / n_ans if n_ans else 0.0,
        "abstention": abstain_ok / n_out if n_out else 0.0,
        "answered": answered / n_ans if n_ans else 0.0,
        "n_answerable": n_ans,
        "n_out": n_out,
    }


def main():
    items = load_items()
    retriever = Retriever()
    k, min_sim = config.TOP_K, config.MIN_SIMILARITY
    runs = collect(retriever, items, k)

    for run in runs:
        it = run["item"]
        if it["answerable"]:
            rank = first_hit_rank(it, run["results"])
            tag = "HIT " if rank else "MISS"
            print(f"[{tag}] rank={rank} best_sim={run['best_similarity']:.2f} | {it['question']}")
        else:
            refused = run["best_similarity"] < min_sim
            print(
                f"[{'OK  ' if refused else 'FAIL'}] best_sim={run['best_similarity']:.2f} "
                f"| {it['question']}"
            )

    s = score(runs, k, min_sim)
    print(
        f"\n=== Results (TOP_K={k}, MIN_SIMILARITY={min_sim}, CHUNK_SIZE={config.CHUNK_SIZE}, "
        f"CHUNK_OVERLAP={config.CHUNK_OVERLAP}, USE_RERANKER={config.USE_RERANKER}) ==="
    )
    print(f"Hit@{k}: {s['hit'] * s['n_answerable']:.0f}/{s['n_answerable']} = {s['hit']:.0%}")
    print(f"MRR: {s['mrr']:.3f}")
    print(f"Abstention: {s['abstention'] * s['n_out']:.0f}/{s['n_out']} = {s['abstention']:.0%}")
    print(f"Answerable questions answered (not refused): {s['answered']:.0%}")


if __name__ == "__main__":
    main()
