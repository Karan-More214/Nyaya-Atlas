"""Grid-search retrieval settings against the current index.

Usage (index must already be built with the chunk settings you want to test):
    python -m eval.sweep

Retrieval is run once per question; TOP_K and MIN_SIMILARITY are then applied to the
saved results, so the sweep is cheap. Chunk settings change the index, so re-ingest
for each one (CHUNK_SIZE=800 python -m src.ingest) and run the sweep again.
Prints one JSON object so a driver script can collect several runs.
"""
import json
import sys

from src import config
from src.retriever import Retriever
from eval.run_eval import collect, load_items, score

TOP_KS = [3, 5, 8]
THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main():
    items = load_items()
    retriever = Retriever()
    out = {
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "n_chunks": len(retriever.chunks),
        "retrieval": {},
        "threshold": {},
    }

    for rerank in (False, True):
        if rerank and retriever.reranker is None:
            from sentence_transformers import CrossEncoder

            retriever.reranker = CrossEncoder(config.RERANK_MODEL)
        if not rerank:
            retriever.reranker = None
        runs = collect(retriever, items, k=max(TOP_KS))
        for k in TOP_KS:
            s = score(runs, k, 0.0)
            out["retrieval"][f"rerank={rerank},k={k}"] = {
                "hit": round(s["hit"], 3),
                "mrr": round(s["mrr"], 3),
            }
        if not rerank:
            for t in THRESHOLDS:  # best_similarity is vector-only, so same for both
                s = score(runs, 5, t)
                out["threshold"][str(t)] = {
                    "abstention": round(s["abstention"], 3),
                    "answered": round(s["answered"], 3),
                }
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
