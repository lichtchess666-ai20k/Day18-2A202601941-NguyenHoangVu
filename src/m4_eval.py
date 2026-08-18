from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _as_score(value) -> float:
    """Ép về float; NaN (RAGAS trả khi không chấm được) → 0.0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if score != score else score          # NaN != NaN


def _as_list(value) -> list:
    """contexts từ RAGAS là numpy array — KHÔNG dùng `or` lên nó,
    bool(array nhiều phần tử) ném ValueError."""
    if value is None:
        return []
    return [str(item) for item in value]


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    empty = {name: 0.0 for name in _METRIC_NAMES}
    empty["per_question"] = []

    # RAGAS cần OPENAI_API_KEY và Python 3.11+ (asyncio). Bọc try/except để
    # pipeline vẫn chạy hết và ghi được report kể cả khi eval hỏng.
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                contexts=_as_list(row.get("contexts")),
                ground_truth=row.get("ground_truth", ""),
                faithfulness=_as_score(row.get("faithfulness")),
                answer_relevancy=_as_score(row.get("answer_relevancy")),
                context_precision=_as_score(row.get("context_precision")),
                context_recall=_as_score(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]

        results = {}
        for name in _METRIC_NAMES:
            values = [getattr(item, name) for item in per_question]
            results[name] = sum(values) / len(values) if values else 0.0
        results["per_question"] = per_question
        return results

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return empty


# Diagnostic Tree: metric thấp nhất → chẩn đoán nguyên nhân → hướng sửa
DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM bịa — câu trả lời chứa thông tin không có trong context",
        "Siết prompt (chỉ trả lời từ context), hạ temperature, bắt trích dẫn nguồn",
    ),
    "context_recall": (
        "Retrieval bỏ sót chunk chứa đáp án",
        "Chỉnh lại chunking, tăng top_k, hoặc thêm BM25 vào hybrid search",
    ),
    "context_precision": (
        "Context lẫn quá nhiều chunk nhiễu, chunk đúng bị đẩy xuống dưới",
        "Thêm reranking hoặc lọc theo metadata (phiên bản, ngày hiệu lực)",
    ),
    "answer_relevancy": (
        "Câu trả lời lạc đề so với câu hỏi",
        "Sửa prompt template, yêu cầu trả lời trực tiếp và đúng trọng tâm",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analyzed = []

    for item in eval_results:
        scores = {name: _as_score(getattr(item, name, 0.0)) for name in _METRIC_NAMES}
        worst_metric = min(scores, key=lambda name: scores[name])
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]

        analyzed.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "all_scores": {name: round(value, 4) for name, value in scores.items()},
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    analyzed.sort(key=lambda entry: entry["avg_score"])      # tệ nhất lên đầu
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
