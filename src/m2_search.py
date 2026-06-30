from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import math
import os, sys
import re
from collections import Counter
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


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _hash_vector(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    counts = Counter(_tokens(text))
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * count
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
    except Exception:
        segmented = text
    # 1. from underthesea import word_tokenize
    # 2. segmented = word_tokenize(text, format="text")
    # 3. return segmented.replace("_", " ")
    #
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        # 1. self.documents = chunks
        # 2. For each chunk: segment_vietnamese(chunk["text"]) → split by space
        # 3. self.corpus_tokens = [tokenized list for each chunk]
        # 4. from rank_bm25 import BM25Okapi
        #    self.bm25 = BM25Okapi(self.corpus_tokens)
        self.corpus_tokens = [_tokens(segment_vietnamese(chunk["text"])) for chunk in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        # 1. if self.bm25 is None: return []
        # 2. tokenized_query = segment_vietnamese(query).split()
        # 3. scores = self.bm25.get_scores(tokenized_query)
        # 4. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 5. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        #    Lọc scores[i] > 0 để bỏ docs không liên quan.
        tokenized_query = _tokens(segment_vietnamese(query))
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc["text"],
                score=score,
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
        except Exception:
            self.client = None
        self._encoder = None
        self._qdrant_ready = False
        self.documents: list[dict] = []
        self.vectors: list[list[float]] = []

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                allow_download = os.getenv("LAB18_ALLOW_MODEL_DOWNLOAD", "0") == "1"
                self._encoder = SentenceTransformer(
                    EMBEDDING_MODEL,
                    local_files_only=not allow_download,
                )
            except Exception:
                self._encoder = False
        return self._encoder

    def _encode(self, texts: list[str]) -> list[list[float]]:
        encoder = self._get_encoder()
        if encoder:
            vectors = encoder.encode(texts, normalize_embeddings=True)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
        return [_hash_vector(text) for text in texts]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self.documents = chunks
        # 1. from qdrant_client.models import Distance, VectorParams, PointStruct
        # 2. self.client.recreate_collection(collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        # 3. texts = [c["text"] for c in chunks]
        # 4. vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        # 5. points = [PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]}) ...]
        # 6. self.client.upsert(collection, points)
        texts = [c["text"] for c in chunks]
        self.vectors = self._encode(texts) if texts else []
        self._qdrant_ready = False

        if not self.client or not self.vectors:
            return

        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=len(self.vectors[0]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector,
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
                )
                for i, vector in enumerate(self.vectors)
            ]
            self.client.upsert(collection_name=collection, points=points)
            self._qdrant_ready = True
        except Exception as exc:
            print(f"  Warning: Qdrant unavailable, using in-memory dense search ({exc})", flush=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not self.documents:
            return []
        # 1. query_vector = self._get_encoder().encode(query).tolist()
        # 2. response = self.client.query_points(collection, query=query_vector, limit=top_k)
        # 3. Return [SearchResult(text=pt.payload["text"], score=pt.score, metadata=pt.payload, method="dense")
        #            for pt in response.points]
        #
        # ⚠️ LƯU Ý: qdrant-client >= 2.0 dùng query_points(), KHÔNG phải search().
        query_vector = self._encode([query])[0]

        if self.client and self._qdrant_ready:
            try:
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception as exc:
                print(f"  Warning: Qdrant search failed, using in-memory dense search ({exc})", flush=True)
                self._qdrant_ready = False

        scored = sorted(
            ((_cosine(query_vector, vector), i) for i, vector in enumerate(self.vectors)),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(score),
                metadata=self.documents[i].get("metadata", {}),
                method="dense",
            )
            for score, i in scored[:top_k]
            if score > 0
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, dict] = {}
    # 1. rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    # 2. For each result_list in results_list:
    #      For rank, result in enumerate(result_list):
    #        if result.text not in rrf_scores: rrf_scores[result.text] = {"score": 0.0, "result": result}
    #        rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    # 3. Sort by score descending
    # 4. Return top_k SearchResult with method="hybrid"
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            item = fused.setdefault(result.text, {"score": 0.0, "result": result})
            item["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked
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


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
