from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, sys
import re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, create_llm_client


_LLM_DISABLED = False


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _llm_enabled() -> bool:
    return bool(LLM_API_KEY) and os.getenv("LAB18_USE_LLM_ENRICHMENT", "0") == "1" and not _LLM_DISABLED


def _call_llm(messages: list[dict], max_tokens: int = 300) -> str:
    global _LLM_DISABLED
    if not _llm_enabled():
        return ""
    try:
        client = create_llm_client()
        if client is None:
            return ""
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  Warning: enrichment LLM disabled after failure ({exc})", flush=True)
        _LLM_DISABLED = True
        return ""


def _json_from_text(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _fallback_summary(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:2]).strip()


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    summary = _fallback_summary(text)
    base = summary[:120].strip()
    questions = [
        "Thông tin chính trong đoạn này là gì?",
        "Đoạn này áp dụng cho trường hợp nào?",
        f"Nội dung nào liên quan đến {base}?" if base else "Đoạn này trả lời câu hỏi nào?",
    ]
    return questions[:n_questions]


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    if any(token in lowered for token in ["vpn", "mật khẩu", "password", "aes", "wireguard"]):
        category = "it"
    elif any(token in lowered for token in ["lương", "phụ cấp", "chi phí", "tạm ứng", "thưởng"]):
        category = "finance"
    elif any(token in lowered for token in ["nghỉ", "thử việc", "đào tạo", "mentor", "nhân viên"]):
        category = "hr"
    else:
        category = "policy"
    topic = _fallback_summary(text)[:80] or "general"
    return {"topic": topic, "entities": [], "category": category, "language": "vi"}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    content = _call_llm([
        {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2 câu ngắn gọn bằng tiếng Việt."},
        {"role": "user", "content": text},
    ], max_tokens=150)
    if content:
        return content
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return resp.choices[0].message.content.strip()
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI summarize failed: {e}")
    #
    # Extractive fallback (không cần API):
    # sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    # return ". ".join(sentences[:2]) + "." if sentences else text
    return _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    content = _call_llm([
        {"role": "system", "content": f"Tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu trên một dòng."},
        {"role": "user", "content": text},
    ], max_tokens=200)
    if content:
        questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in content.splitlines()
            if q.strip()
        ]
        if questions:
            return questions[:n_questions]
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=200,
    #         )
    #         questions = resp.choices[0].message.content.strip().split("\n")
    #         return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI HyQA failed: {e}")
    #
    # Extractive fallback:
    # import re
    # sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    # return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    content = _call_llm([
        {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì."},
        {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=80)
    if content:
        return f"{content}\n\n{text}"
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
    #                 {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=80,
    #         )
    #         context = resp.choices[0].message.content.strip()
    #         return f"{context}\n\n{text}"
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI contextual failed: {e}")
    #
    # Simple fallback:
    # prefix = f"Trích từ {document_title}. " if document_title else ""
    # return f"{prefix}{text}"
    prefix = f"Trích từ {document_title}. " if document_title else "Ngữ cảnh tài liệu: "
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    content = _call_llm([
        {"role": "system", "content": 'Trích xuất metadata dạng JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
        {"role": "user", "content": text},
    ], max_tokens=150)
    metadata = _json_from_text(content)
    if metadata:
        return metadata
    # if OPENAI_API_KEY:
    #     try:
    #         import json as _json
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI metadata failed: {e}")
    #
    # return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    content = _call_llm([
        {"role": "system", "content": """Phân tích đoạn văn và trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
        {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    ], max_tokens=400)
    data = _json_from_text(content)
    if data:
        return data
    # if OPENAI_API_KEY:
    #     try:
    #         import json as _json
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
    # {
    #   "summary": "tóm tắt 2-3 câu",
    #   "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
    #   "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
    #   "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
    # }"""},
    #                 {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=400,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  Enrichment API failed: {e}")
    context = f"Trích từ {source}." if source else "Ngữ cảnh tài liệu."
    return {
        "summary": _fallback_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": context,
        "metadata": _fallback_metadata(text),
    }


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
