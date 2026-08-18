from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
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
    except Exception:
        return text                      # fallback: giữ nguyên nếu underthesea lỗi
    # underthesea nối từ ghép bằng "_" ("nghỉ_phép"). BM25 tokenize bằng split(" "),
    # nên phải trả "_" về khoảng trắng, nếu không query "nghỉ phép" (2 token)
    # sẽ không bao giờ khớp document "nghỉ_phép" (1 token).
    return segmented.replace("_", " ")


def _tokenize(text: str) -> list[str]:
    """Chuẩn hoá 1 lần cho cả document lẫn query — phải dùng chung để token khớp nhau."""
    return segment_vietnamese(text).lower().split()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks or []
        self.corpus_tokens = [_tokenize(c["text"]) for c in self.documents]
        # BM25Okapi ném ZeroDivisionError nếu corpus rỗng
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in ranked:
            if scores[i] <= 0:           # bỏ document không chia sẻ token nào với query
                continue
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
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

        if not chunks:
            return

        # recreate_collection() đã deprecated ở qdrant-client 1.19 → dùng delete + create
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, batch_size=8)

        points = [
            PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={**chunks[i].get("metadata", {}), "text": texts[i]},
            )
            for i, vector in enumerate(vectors)
        ]
        for start in range(0, len(points), 64):      # upsert theo batch cho corpus lớn
            self.client.upsert(collection_name=collection, points=points[start:start + 64])

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            # qdrant-client >= 1.9 bỏ hẳn .search() — chỉ còn query_points()
            response = self.client.query_points(
                collection_name=collection, query=query_vector, limit=top_k,
            )
        except Exception as e:
            print(f"  ⚠️  Dense search failed: {e}")
            return []

        return [
            SearchResult(
                text=point.payload.get("text", ""),
                score=float(point.score),
                metadata={k: v for k, v in point.payload.items() if k != "text"},
                method="dense",
            )
            for point in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    fused: dict[str, dict] = {}

    for results in results_list:
        for rank, result in enumerate(results):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            # +1 vì enumerate đếm từ 0, còn công thức RRF dùng rank bắt đầu từ 1
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=entry["result"].text,
            score=entry["score"],
            metadata=entry["result"].metadata,
            method="hybrid",
        )
        for entry in ranked
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
