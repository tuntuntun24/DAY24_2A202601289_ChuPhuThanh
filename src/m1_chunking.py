from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

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


@lru_cache(maxsize=1)
def _get_semantic_model():
    """Load the embedding model once instead of once per document."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _lexical_similarity(left: str, right: str) -> float:
    """Dependency-free similarity used when the embedding model is unavailable."""
    left_tokens = set(re.findall(r"\w+", left.lower(), flags=re.UNICODE))
    right_tokens = set(re.findall(r"\w+", right.lower(), flags=re.UNICODE))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _attach_headers_to_content(parts: list[str]) -> list[str]:
    """Keep Markdown headings with the first sentence of their section."""
    result: list[str] = []
    pending_headers: list[str] = []
    for part in parts:
        if re.fullmatch(r"#{1,6}\s+.+", part):
            pending_headers.append(part)
            continue
        if pending_headers:
            part = "\n".join([*pending_headers, part])
            pending_headers = []
        result.append(part)
    if pending_headers:
        result.append("\n".join(pending_headers))
    return result


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = _attach_headers_to_content([
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
        if sentence.strip()
    ])
    if not sentences:
        return []

    try:
        import numpy as np

        embeddings = _get_semantic_model().encode(sentences)

        def similarity(index: int) -> float:
            left = embeddings[index - 1]
            right = embeddings[index]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right)) + 1e-9
            return float(np.dot(left, right) / denominator)

    except (ImportError, OSError, RuntimeError) as exc:
        # The lab should still work offline or before the optional model is cached.
        print(f"  ⚠️  Semantic model unavailable ({exc}); using lexical fallback.")

        def similarity(index: int) -> float:
            return _lexical_similarity(sentences[index - 1], sentences[index])

    groups: list[list[str]] = [[sentences[0]]]
    for index in range(1, len(sentences)):
        if similarity(index) < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_to_size(text: str, max_size: int) -> list[str]:
    """Split text on natural boundaries while keeping every part within max_size."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")

    units = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    pieces: list[str] = []
    for unit in units:
        if len(unit) <= max_size:
            pieces.append(unit)
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", unit)
            if sentence.strip()
        ]
        for sentence in sentences:
            if len(sentence) <= max_size:
                pieces.append(sentence)
                continue
            # A single oversized sentence is split by words, with a hard split as
            # a last resort for tokens longer than max_size.
            words = re.findall(r"\S+", sentence)
            current = ""
            for word in words:
                if len(word) > max_size:
                    if current:
                        pieces.append(current)
                        current = ""
                    pieces.extend(word[i:i + max_size] for i in range(0, len(word), max_size))
                elif not current:
                    current = word
                elif len(current) + 1 + len(word) <= max_size:
                    current += " " + word
                else:
                    pieces.append(current)
                    current = word
            if current:
                pieces.append(current)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(piece) > max_size:
            chunks.append(current)
            current = piece
        else:
            current += separator + piece
    if current:
        chunks.append(current)
    return chunks


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if child_size <= 0 or parent_size <= 0:
        raise ValueError("parent_size and child_size must be greater than zero")
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")

    metadata = metadata or {}
    parent_texts = _split_to_size(text, parent_size)
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(
            Chunk(
                text=parent_text,
                metadata={
                    **metadata,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                    "chunk_index": parent_index,
                },
            )
        )
        for child_text in _split_to_size(parent_text, child_size):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                        "chunk_index": len(children),
                    },
                    parent_id=parent_id,
                )
            )

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^#{1,3}\s+.+$", flags=re.MULTILINE)
    matches = list(header_pattern.finditer(text))

    # Plain text is still a valid single logical section.
    if not matches:
        stripped = text.strip()
        if not stripped:
            return []
        return [
            Chunk(
                text=stripped,
                metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0},
            )
        ]

    chunks: list[Chunk] = []
    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(
            Chunk(
                text=preamble,
                metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0},
            )
        )

    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start():section_end].strip()
        header = match.group(0).strip()
        if section_text:
            chunks.append(
                Chunk(
                    text=section_text,
                    metadata={
                        **metadata,
                        "section": header.lstrip("#").strip(),
                        "header": header,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )

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

