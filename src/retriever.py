"""Hybrid retrieval: BM25 keyword search + vector search, fused with
Reciprocal Rank Fusion (RRF), with an optional cross-encoder reranker."""
import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        if not config.CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Index not found. Put documents in data/raw and run: python -m src.ingest"
            )
        with open(config.CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.by_id = {c["id"]: c for c in self.chunks}

        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(config.COLLECTION_NAME)

        self.reranker = None
        if config.USE_RERANKER:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(config.RERANK_MODEL)

    # ---- individual retrievers -------------------------------------------
    def _vector(self, query: str, n: int):
        n = max(1, min(n, len(self.chunks)))  # Chroma errors if n > collection size
        emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=n)
        ids = res["ids"][0]
        sims = [1.0 - d for d in res["distances"][0]]  # cosine distance -> similarity
        return ids, sims

    def _keyword(self, query: str, n: int):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i]["id"] for i in order if scores[i] > 0]

    # ---- public API -------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> dict:
        k = k or config.TOP_K
        n = config.CANDIDATES
        if not query or not query.strip():
            return {"results": [], "best_similarity": 0.0}
        vec_ids, vec_sims = self._vector(query, n)
        kw_ids = self._keyword(query, n)
        sim_by_id = dict(zip(vec_ids, vec_sims))

        # Reciprocal Rank Fusion
        fused: dict[str, float] = {}
        for ranking in (vec_ids, kw_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)

        if self.reranker is not None:
            pool = ranked[: max(k * 3, 10)]
            scores = self.reranker.predict([(query, self.by_id[c]["text"]) for c in pool])
            order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
            ranked = [pool[i] for i in order]

        results = []
        for cid in ranked[:k]:
            c = self.by_id[cid]
            results.append(
                {
                    "id": cid,
                    "text": c["text"],
                    "source": c["source"],
                    "title": c["title"],
                    "page": c["page"],
                    "similarity": sim_by_id.get(cid),
                }
            )
        return {
            "results": results,
            "best_similarity": max(vec_sims) if vec_sims else 0.0,
        }
