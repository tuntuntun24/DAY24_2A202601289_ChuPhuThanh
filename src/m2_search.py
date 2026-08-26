from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text or not text.strip():
        return ""

    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except (ImportError, RuntimeError):
        # Keep BM25 usable in a minimal/offline environment.
        segmented = text

    # underthesea joins compound words with underscores. Replacing them keeps
    # document and query tokenization consistent with the lab specification.
    return re.sub(r"\s+", " ", segmented.replace("_", " ")).strip().lower()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = []
        for index, chunk in enumerate(self.documents):
            if "text" not in chunk:
                raise ValueError(f"Chunk at index {index} is missing 'text'")
            self.corpus_tokens.append(segment_vietnamese(str(chunk["text"])).split())

        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(
                SearchResult(
                    text=str(document["text"]),
                    score=score,
                    metadata=dict(document.get("metadata") or {}),
                    method="bm25",
                )
            )
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        documents = list(chunks)
        for index, chunk in enumerate(documents):
            if "text" not in chunk:
                raise ValueError(f"Chunk at index {index} is missing 'text'")

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not documents:
            return

        texts = [str(chunk["text"]) for chunk in documents]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = [
            PointStruct(
                id=index,
                vector=vector.tolist(),
                payload={**dict(chunk.get("metadata") or {}), "text": texts[index]},
            )
            for index, (chunk, vector) in enumerate(zip(documents, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip():
            return []

        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            results.append(
                SearchResult(
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    metadata=payload,
                    method="dense",
                )
            )
        return results

    def search_batch(self, queries: list[str], top_k: int = DENSE_TOP_K,
                     collection: str = COLLECTION_NAME) -> list[list[SearchResult]]:
        """Encode many queries in one batch, then query Qdrant per vector."""
        if top_k <= 0:
            return [[] for _ in queries]
        nonempty = [(index, query) for index, query in enumerate(queries) if query.strip()]
        output: list[list[SearchResult]] = [[] for _ in queries]
        if not nonempty:
            return output
        vectors = self._get_encoder().encode([query for _, query in nonempty], show_progress_bar=True)
        for (index, _query), vector in zip(nonempty, vectors):
            response = self.client.query_points(
                collection_name=collection,
                query=vector.tolist(),
                limit=top_k,
                with_payload=True,
            )
            output[index] = [
                SearchResult(
                    text=str((point.payload or {}).get("text", "")),
                    score=float(point.score),
                    metadata=dict(point.payload or {}),
                    method="dense",
                )
                for point in response.points
            ]
        return output


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    fused: dict[str, dict] = {}
    for result_list in results_list:
        seen_in_list: set[str] = set()
        for rank, result in enumerate(result_list):
            # A duplicated result in one retriever should not vote twice.
            if result.text in seen_in_list:
                continue
            seen_in_list.add(result.text)
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda entry: entry["score"], reverse=True)
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)

    def search_batch(self, queries: list[str], top_k: int = HYBRID_TOP_K) -> list[list[SearchResult]]:
        """Hybrid retrieval with batched dense encoding for production evaluation."""
        bm25_batches = [self.bm25.search(query, top_k=BM25_TOP_K) for query in queries]
        dense_batches = self.dense.search_batch(queries, top_k=DENSE_TOP_K)
        return [
            reciprocal_rank_fusion([bm25, dense], top_k=top_k)
            for bm25, dense in zip(bm25_batches, dense_batches)
        ]


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
