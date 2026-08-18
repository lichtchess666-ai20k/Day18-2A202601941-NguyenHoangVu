from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_SENTENCE_SPLIT = r'(?<=[.!?])\s+|\n\n'
_semantic_model = None


def _get_semantic_model():
    """Cache model ở module level — tránh load lại mỗi lần gọi chunk_semantic()."""
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def _split_sentences(text: str) -> list[str]:
    """Tách câu theo dấu kết thúc câu (. ! ?) hoặc xuống dòng kép."""
    return [s.strip() for s in re.split(_SENTENCE_SPLIT, text) if s.strip()]


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    def _make(group: list[str], index: int) -> Chunk:
        return Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "chunk_index": index, "strategy": "semantic"},
        )

    if len(sentences) == 1:
        return [_make(sentences, 0)]

    embeddings = _get_semantic_model().encode(sentences)

    chunks: list[Chunk] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        prev, curr = embeddings[i - 1], embeddings[i]
        similarity = float(dot(prev, curr) / (norm(prev) * norm(curr) + 1e-9))
        if similarity < threshold:
            # Đổi chủ đề → chốt chunk hiện tại, mở chunk mới
            chunks.append(_make(current, len(chunks)))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    chunks.append(_make(current, len(chunks)))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _pack_to_size(units: list[str], size: int, joiner: str = " ") -> list[str]:
    """Gộp các đơn vị text thành nhóm, mỗi nhóm ≤ size ký tự."""
    packed: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > size:              # đơn vị dài hơn size → buộc phải cắt cứng
            if current:
                packed.append(current.strip())
                current = ""
            packed.append(unit[:size].strip())
            unit = unit[size:]
        if current and len(current) + len(joiner) + len(unit) > size:
            packed.append(current.strip())
            current = ""
        current = f"{current}{joiner}{unit}" if current else unit
    if current.strip():
        packed.append(current.strip())
    return packed


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    def _add_parent(buffer: list[str]) -> None:
        """Chốt 1 parent từ buffer paragraph, rồi cắt nhỏ thành children."""
        if not buffer:
            return
        pid = f"parent_{len(parents)}"
        parent_text = "\n\n".join(buffer).strip()
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "chunk_index": len(parents), "strategy": "hierarchical"},
        ))
        for child_text in _pack_to_size(_split_sentences(parent_text), child_size):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "chunk_index": len(children), "strategy": "hierarchical"},
                parent_id=pid,
            ))

    buffer: list[str] = []
    buffer_len = 0
    for para in paragraphs:
        if buffer and buffer_len + len(para) > parent_size:
            _add_parent(buffer)
            buffer, buffer_len = [], 0
        buffer.append(para)
        buffer_len += len(para) + 2         # +2 cho "\n\n" nối giữa các paragraph
    _add_parent(buffer)

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────

_HEADER_PATTERN = r'(^#{1,3}\s+.+$)'


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    # re.split có capture group → [phần trước header, header, nội dung, header, nội dung, ...]
    parts = re.split(_HEADER_PATTERN, text, flags=re.MULTILINE)

    chunks: list[Chunk] = []

    def _add(header: str, body: str) -> None:
        section = header.lstrip("#").strip()
        full = f"{header}\n\n{body.strip()}".strip() if header else body.strip()
        if not full:
            return
        chunks.append(Chunk(
            text=full,
            metadata={**metadata, "section": section or "(preamble)",
                      "strategy": "structure", "chunk_index": len(chunks)},
        ))

    _add("", parts[0])                      # nội dung đứng trước header đầu tiên
    for i in range(1, len(parts), 2):       # chỉ số lẻ = header, kế tiếp = nội dung của nó
        _add(parts[i].strip(), parts[i + 1] if i + 1 < len(parts) else "")

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
