from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import math
import os
import re
import sys
import time
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    _model_cache: ClassVar[dict[str, object]] = {}
    _unavailable_models: ClassVar[set[str]] = set()

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if not self.model_name or self.model_name == "lexical":
            return None
        if self._model is not None:
            return self._model
        if self.model_name in self._model_cache:
            self._model = self._model_cache[self.model_name]
            return self._model
        if self.model_name in self._unavailable_models:
            return None

        try:
            # FlagEmbedding is intentionally avoided because its tokenizer is
            # incompatible with some newer transformers releases.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._model_cache[self.model_name] = self._model
        except Exception as exc:
            self._unavailable_models.add(self.model_name)
            print(f"  ⚠️  Reranker model unavailable ({exc}); using lexical fallback.")
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []

        normalized_documents: list[dict] = []
        for index, document in enumerate(documents):
            if "text" not in document:
                raise ValueError(f"Document at index {index} is missing 'text'")
            normalized_documents.append(document)

        model = self._load_model()
        if model is None:
            scores = [_lexical_relevance(query, str(doc["text"])) for doc in normalized_documents]
        else:
            pairs = [(query, str(doc["text"])) for doc in normalized_documents]
            raw_scores = model.predict(pairs)
            if isinstance(raw_scores, Real):
                scores = [float(raw_scores)]
            else:
                # CrossEncoder normally returns a numpy array. Flattening also
                # supports the (n, 1) shape returned by some model versions.
                try:
                    import numpy as np

                    scores = [float(score) for score in np.asarray(raw_scores).reshape(-1)]
                except ImportError:
                    scores = [float(score) for score in raw_scores]

        if len(scores) != len(normalized_documents):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for "
                f"{len(normalized_documents)} documents"
            )

        scored = sorted(
            zip(scores, normalized_documents),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            RerankResult(
                text=str(document["text"]),
                original_score=float(document.get("score") or 0.0),
                rerank_score=float(score),
                metadata=dict(document.get("metadata") or {}),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Optional lightweight extension; CrossEncoderReranker is the required
        # production implementation for this lab.
        return []


def _lexical_relevance(query: str, document: str) -> float:
    """Cosine similarity over term frequencies for an offline fallback."""
    query_tokens = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    document_tokens = re.findall(r"\w+", document.lower(), flags=re.UNICODE)
    if not query_tokens or not document_tokens:
        return 0.0

    query_counts = {token: query_tokens.count(token) for token in set(query_tokens)}
    document_counts = {token: document_tokens.count(token) for token in set(document_tokens)}
    dot_product = sum(
        count * document_counts.get(token, 0)
        for token, count in query_counts.items()
    )
    query_norm = math.sqrt(sum(count * count for count in query_counts.values()))
    document_norm = math.sqrt(sum(count * count for count in document_counts.values()))
    return dot_product / (query_norm * document_norm) if query_norm and document_norm else 0.0


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be greater than zero")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
