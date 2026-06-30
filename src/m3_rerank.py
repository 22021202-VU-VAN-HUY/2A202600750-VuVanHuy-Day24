from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import math
import os, sys, time
import re
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


_MODEL_CACHE = {}


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _tokens(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _lexical_score(query: str, document: str, original_score: float = 0.0) -> float:
    q = _tokens(query)
    d = _tokens(document)
    if not q or not d:
        return float(original_score) * 0.01
    overlap = set(q) & set(d)
    weighted_overlap = sum(min(q[token], d[token]) for token in overlap)
    q_norm = math.sqrt(sum(v * v for v in q.values()))
    d_norm = math.sqrt(sum(v * v for v in d.values()))
    cosine = weighted_overlap / (q_norm * d_norm + 1e-9)
    number_bonus = 0.15 * len(set(re.findall(r"\d+", query)) & set(re.findall(r"\d+", document)))
    return cosine + number_bonus + float(original_score) * 0.01


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if self.model_name in _MODEL_CACHE:
                self._model = _MODEL_CACHE[self.model_name]
                return self._model
            try:
                from sentence_transformers import CrossEncoder
                allow_download = os.getenv("LAB18_ALLOW_MODEL_DOWNLOAD", "0") == "1"
                self._model = CrossEncoder(
                    self.model_name,
                    local_files_only=not allow_download,
                    max_length=512,
                )
            except Exception:
                self._model = False
            _MODEL_CACHE[self.model_name] = self._model
            # from sentence_transformers import CrossEncoder
            # self._model = CrossEncoder(self.model_name)
            #
            # ⚠️ LƯU Ý: Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        # 1. if not documents: return []
        # 2. model = self._load_model()
        # 3. pairs = [(query, doc["text"]) for doc in documents]
        # 4. scores = model.predict(pairs)
        # 5. if isinstance(scores, (int, float)): scores = [scores]
        # 6. scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        # 7. Return [RerankResult(text=..., original_score=doc.get("score", 0.0),
        #            rerank_score=float(score), metadata=..., rank=i)
        #            for i, (score, doc) in enumerate(scored[:top_k])]
        model = self._load_model()
        if model:
            pairs = [(query, doc["text"]) for doc in documents]
            scores = model.predict(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]
            elif hasattr(scores, "tolist"):
                scores = scores.tolist()
        else:
            scores = [
                _lexical_score(query, doc["text"], doc.get("score", 0.0))
                for doc in documents
            ]

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Optional fast path: from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
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
