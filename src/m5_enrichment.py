from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

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


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    fallback = _extractive_summary(text)
    if not OPENAI_API_KEY or not text.strip():
        return fallback

    try:
        return _chat_completion(
            "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
            "Không thêm thông tin không có trong đoạn văn.",
            text,
            max_tokens=150,
        ) or fallback
    except Exception as exc:
        print(f"  [warning] OpenAI summarize failed: {exc}")
        return fallback


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    fallback = _fallback_questions(text, n_questions)
    if not OPENAI_API_KEY or not text.strip():
        return fallback

    try:
        content = _chat_completion(
            f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
            "Mỗi câu hỏi trên một dòng, không kèm lời giải thích.",
            text,
            max_tokens=200,
        )
        questions = _clean_questions(content, n_questions)
        return questions or fallback
    except Exception as exc:
        print(f"  [warning] OpenAI HyQA failed: {exc}")
        return fallback


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    fallback_context = _fallback_context(text, document_title)
    if not OPENAI_API_KEY or not text.strip():
        return _prepend_context(text, fallback_context)

    try:
        context = _chat_completion(
            "Viết đúng một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu "
            "và nói về chủ đề gì. Không thêm thông tin mới.",
            f"Tài liệu: {document_title or 'Không rõ'}\n\nĐoạn văn:\n{text}",
            max_tokens=80,
        )
        return _prepend_context(text, context or fallback_context)
    except Exception as exc:
        print(f"  [warning] OpenAI contextual failed: {exc}")
        return _prepend_context(text, fallback_context)


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    fallback = _fallback_metadata(text)
    if not OPENAI_API_KEY or not text.strip():
        return fallback

    try:
        content = _chat_completion(
            'Trích xuất metadata và chỉ trả về JSON hợp lệ dạng '
            '{"topic":"...","entities":["..."],'
            '"category":"policy|hr|it|finance","language":"vi|en"}.',
            text,
            max_tokens=150,
            json_mode=True,
        )
        return _normalize_metadata(_parse_json_object(content), fallback)
    except Exception as exc:
        print(f"  [warning] OpenAI metadata failed: {exc}")
        return fallback


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    fallback = _fallback_enrichment(text, source)
    if not OPENAI_API_KEY or not text.strip():
        return fallback

    try:
        content = _chat_completion(
            """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "một câu mô tả vị trí và chủ đề của đoạn văn",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}
Không thêm dữ kiện ngoài đoạn văn.""",
            f"Tài liệu: {source or 'Không rõ'}\n\nĐoạn văn:\n{text}",
            max_tokens=400,
            json_mode=True,
        )
        result = _parse_json_object(content)
        return {
            "summary": str(result.get("summary") or fallback["summary"]).strip(),
            "questions": _normalize_questions(result.get("questions"), fallback["questions"]),
            "context": str(result.get("context") or fallback["context"]).strip(),
            "metadata": _normalize_metadata(result.get("metadata"), fallback["metadata"]),
        }
    except Exception as exc:
        print(f"  [warning] Enrichment API failed: {exc}")
        return fallback


def _chat_completion(system_prompt: str, user_content: str, max_tokens: int,
                     json_mode: bool = False) -> str:
    """Call OpenAI using one small, consistent wrapper."""
    from openai import OpenAI

    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _extractive_summary(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip() and not re.fullmatch(r"#{1,6}\s+.+", sentence.strip())
    ]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    sentences = [
        sentence.strip().rstrip(".!?")
        for sentence in re.split(r"[.!?\n]+", text)
        if len(sentence.strip()) > 10 and not sentence.lstrip().startswith("#")
    ]
    questions = []
    for sentence in sentences[:n_questions]:
        numeric_question = re.sub(
            r"\b\d+(?:[.,]\d+)*\b",
            "bao nhiêu",
            sentence,
            count=1,
        )
        questions.append(f"{numeric_question}?")
    return questions


def _clean_questions(content: str, n_questions: int) -> list[str]:
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in content.splitlines()
    ]
    return [line if line.endswith("?") else f"{line}?" for line in lines if line][:n_questions]


def _normalize_questions(value, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    questions = [str(item).strip() for item in value if str(item).strip()]
    return questions[:3] or fallback


def _fallback_context(text: str, source: str) -> str:
    if source:
        return f"Đoạn trích thuộc tài liệu {source}."
    topic = _extractive_summary(text)[:80].rstrip(".!?")
    return f"Đoạn trích này cung cấp thông tin về {topic}." if topic else ""


def _prepend_context(text: str, context: str) -> str:
    return f"{context}\n\n{text}" if context else text


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    category_terms = {
        "it": ("mật khẩu", "vpn", "mfa", "malware", "cntt", "dữ liệu"),
        "finance": ("lương", "chi phí", "phụ cấp", "thanh toán", "tạm ứng", "mua sắm"),
        "hr": ("nhân viên", "nghỉ phép", "thử việc", "đào tạo", "mentor", "buddy"),
    }
    category = next(
        (name for name, terms in category_terms.items() if any(term in lowered for term in terms)),
        "policy",
    )
    entities = re.findall(r"\b(?:[A-ZĐ]{2,}|\d+(?:[.,]\d+)*(?:%|\s*(?:VNĐ|ngày|tháng|năm))?)\b", text)
    language = "vi" if re.search(r"[ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ]", lowered) else "en"
    return {
        "topic": _extractive_summary(text)[:120] or "general",
        "entities": list(dict.fromkeys(entity.strip() for entity in entities))[:10],
        "category": category,
        "language": language,
    }


def _normalize_metadata(value, fallback: dict) -> dict:
    if not isinstance(value, dict):
        return fallback
    category = value.get("category")
    language = value.get("language")
    entities = value.get("entities")
    return {
        "topic": str(value.get("topic") or fallback["topic"]).strip(),
        "entities": [str(item).strip() for item in entities if str(item).strip()]
        if isinstance(entities, list) else fallback["entities"],
        "category": category if category in {"policy", "hr", "it", "finance"} else fallback["category"],
        "language": language if language in {"vi", "en"} else fallback["language"],
    }


def _parse_json_object(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Enrichment response must be a JSON object")
    return value


def _fallback_enrichment(text: str, source: str) -> dict:
    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": _fallback_context(text, source),
        "metadata": _fallback_metadata(text),
    }


def _compose_enriched_text(text: str, context: str = "", summary: str = "",
                           questions: list[str] | None = None) -> str:
    parts = []
    if context:
        parts.append(context)
    if summary:
        parts.append(f"Tóm tắt: {summary}")
    if questions:
        parts.append("Câu hỏi liên quan:\n" + "\n".join(f"- {question}" for question in questions))
    parts.append(f"Nội dung gốc:\n{text}" if parts else text)
    return "\n\n".join(parts)


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

    allowed_methods = {"summary", "hyqa", "contextual", "metadata", "combined"}
    methods = list(dict.fromkeys(methods))
    unknown_methods = set(methods) - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown_methods)}")

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        if "text" not in chunk:
            raise ValueError(f"Chunk at index {i} is missing 'text'")
        text = chunk["text"]
        if not isinstance(text, str):
            raise ValueError(f"Chunk text at index {i} must be a string")
        chunk_metadata = chunk.get("metadata") or {}
        if not isinstance(chunk_metadata, dict):
            raise ValueError(f"Chunk metadata at index {i} must be a dictionary")
        source = str(chunk_metadata.get("source", ""))

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = _compose_enriched_text(text, context_line, summary, questions)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            context_line = _fallback_context(text, source) if "contextual" in methods else ""
            if "contextual" in methods and OPENAI_API_KEY:
                contextual_text = contextual_prepend(text, source)
                context_line = contextual_text[:-len(text)].strip() if text and contextual_text.endswith(text) else ""
            enriched_text = _compose_enriched_text(text, context_line, summary, questions)
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk_metadata, **auto_meta},
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

