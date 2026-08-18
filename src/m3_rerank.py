from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


# Cache model theo tên ở module level: mỗi test khởi tạo CrossEncoderReranker() mới,
# nếu không cache thì bge-reranker-v2-m3 (2.2GB) bị load lại từ đầu cho từng test.
_MODEL_CACHE: dict = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            from sentence_transformers import CrossEncoder

            if self.model_name not in _MODEL_CACHE:
                _MODEL_CACHE[self.model_name] = CrossEncoder(self.model_name)
            self._model = _MODEL_CACHE[self.model_name]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        # Cross-encoder đọc CẢ CẶP (query, doc) cùng lúc → bắt được liên hệ giữa 2 vế,
        # khác bi-encoder vốn encode riêng rồi mới so cosine.
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # key=lambda tránh việc sorted() so sánh tiếp phần tử dict khi 2 score bằng nhau
        scored = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""

    def __init__(self, model_name: str = "ms-marco-MultiBERT-L-12"):
        self.model_name = model_name          # bản multilingual — hỗ trợ tiếng Việt
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []

        from flashrank import RerankRequest

        model = self._load_model()
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        ranked = model.rerank(RerankRequest(query=query, passages=passages))

        results = []
        for rank, item in enumerate(ranked[:top_k]):
            doc = documents[item["id"]]
            results.append(RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(item["score"]),
                metadata=doc.get("metadata", {}),
                rank=rank,
            ))
        return results


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
