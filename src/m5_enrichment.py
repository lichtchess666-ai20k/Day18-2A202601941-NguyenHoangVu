from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── OpenAI helper dùng chung ────────────────────────────

ENRICH_MODEL = "gpt-4o-mini"
_client = None


def _get_client():
    """Tạo client 1 lần rồi tái sử dụng — enrich 100+ chunks nên không dựng lại mỗi lần."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _chat(system: str, user: str, max_tokens: int = 200, as_json: bool = False) -> str:
    """Gọi 1 lượt chat. Trả "" nếu thiếu API key hoặc lỗi → caller tự fallback."""
    if not OPENAI_API_KEY:
        return ""
    kwargs = {}
    if as_json:
        # Bắt buộc: không có response_format thì model hay bọc kết quả trong ```json
        # khiến json.loads() ném JSONDecodeError.
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = _get_client().chat.completions.create(
            model=ENRICH_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI call failed: {e}")
        return ""


def _parse_json(raw: str, default):
    import json as _json
    if not raw:
        return default
    try:
        parsed = _json.loads(raw)
    except Exception as e:
        print(f"  ⚠️  JSON parse failed: {e}")
        return default
    return parsed if isinstance(parsed, dict) else default


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    summary = _chat(
        "Tóm tắt đoạn văn sau bằng tiếng Việt, tối đa 2 câu và PHẢI ngắn hơn đoạn gốc. "
        "Chỉ trả về phần tóm tắt, không thêm lời dẫn.",
        text, max_tokens=150,
    )
    # Summary dài hơn bản gốc là vô nghĩa → quay về extractive
    if summary and len(summary) <= len(text):
        return summary

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return text
    return ". ".join(sentences[:2]).rstrip(".") + "."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    raw = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Mỗi câu hỏi trên 1 dòng, không đánh số, không thêm gì khác.",
        text, max_tokens=200,
    )
    if raw:
        questions = [q.strip().lstrip("0123456789.-) ").strip()
                     for q in raw.split("\n") if q.strip()]
        if questions:
            return questions[:n_questions]

    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết đúng 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu, không thêm gì khác.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
        max_tokens=80,
    )
    if context:
        return f"{context}\n\n{text}"

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    default = {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    raw = _chat(
        'Trích xuất metadata từ đoạn văn, chỉ trả về json hợp lệ theo mẫu: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150, as_json=True,
    )
    return _parse_json(raw, default)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    raw = _chat(
        "Phân tích đoạn văn và chỉ trả về json đúng cấu trúc sau:\n"
        '{\n'
        '  "summary": "tóm tắt 2-3 câu",\n'
        '  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],\n'
        '  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",\n'
        '  "metadata": {"topic": "...", "entities": ["..."], '
        '"category": "policy|hr|it|finance", "language": "vi|en"}\n'
        '}',
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
        max_tokens=400, as_json=True,
    )
    return _parse_json(raw, {})


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
