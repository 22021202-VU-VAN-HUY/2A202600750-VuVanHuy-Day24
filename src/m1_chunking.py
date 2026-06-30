from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import math
import os, sys, glob, re
from collections import Counter
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


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    # 1. from sentence_transformers import SentenceTransformer
    #    from numpy import dot
    #    from numpy.linalg import norm
    # 2. metadata = metadata or {}
    # 3. Split text thành sentences: re.split(r'(?<=[.!?])\s+|\n\n', text)
    # 4. model = SentenceTransformer("all-MiniLM-L6-v2")
    #    embeddings = model.encode(sentences)
    # 5. cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-9)
    # 6. Duyệt từ sentence[1]:
    #      - sim(embedding[i-1], embedding[i]) < threshold → tách chunk mới
    #      - else: gộp vào chunk hiện tại
    # 7. Return [Chunk(text=joined_group, metadata={..., "strategy": "semantic"})]
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if s.strip()
    ]
    if not sentences:
        return []

    def tokenize(value: str) -> Counter:
        return Counter(re.findall(r"\w+", value.lower(), flags=re.UNICODE))

    def cosine(a: Counter, b: Counter) -> float:
        common = set(a) & set(b)
        numerator = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        return numerator / (norm_a * norm_b + 1e-9)

    groups: list[list[str]] = [[sentences[0]]]
    previous = tokenize(sentences[0])
    for sentence in sentences[1:]:
        current = tokenize(sentence)
        if cosine(previous, current) < threshold and groups[-1]:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
        previous = current

    return [
        Chunk(
            text="\n".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
        if "\n".join(group).strip()
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


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
    # 1. metadata = metadata or {}
    # 2. Split text bằng "\n\n" → paragraphs
    # 3. Gộp paragraphs thành parent chunks (mỗi parent ≤ parent_size chars):
    #      pid = f"parent_{len(parents)}"
    #      parents.append(Chunk(text=..., metadata={..., "chunk_type": "parent", "parent_id": pid}))
    # 4. Mỗi parent → split thành children (mỗi child ≤ child_size chars):
    #      children.append(Chunk(text=..., metadata={..., "chunk_type": "child"}, parent_id=pid))
    # 5. return (parents, children)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    def flush_parent(buffer: list[str], parents: list[Chunk]) -> None:
        if not buffer:
            return
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text="\n\n".join(buffer).strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
        ))

    parents: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > parent_size:
            flush_parent(current, parents)
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    flush_parent(current, parents)

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", parent.text) if u.strip()]
        if not units:
            units = [parent.text]

        buffer: list[str] = []
        buffer_len = 0
        for unit in units:
            if buffer and buffer_len + len(unit) + 1 > child_size:
                children.append(Chunk(
                    text=" ".join(buffer).strip(),
                    metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                    parent_id=pid,
                ))
                buffer, buffer_len = [], 0
            buffer.append(unit)
            buffer_len += len(unit) + 1

        if buffer:
            children.append(Chunk(
                text=" ".join(buffer).strip(),
                metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                parent_id=pid,
            ))

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    # 1. metadata = metadata or {}
    # 2. sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    # 3. Duyệt sections:
    #      - Nếu match header (^#{1,3}\s+): lưu header hiện tại, tạo chunk cho content trước đó
    #      - Else: gộp vào content hiện tại
    # 4. Return [Chunk(text=header+content, metadata={..., "section": header, "strategy": "structure"})]
    header_re = re.compile(r"^(#{1,6}\s+.+)$", flags=re.MULTILINE)
    matches = list(header_re.finditer(text))
    if not matches:
        return [
            Chunk(text=c.text, metadata={**c.metadata, **metadata, "strategy": "structure", "section": "document"})
            for c in chunk_basic(text, metadata=metadata)
        ]

    chunks: list[Chunk] = []
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            chunks.append(Chunk(
                text=preamble,
                metadata={**metadata, "strategy": "structure", "section": "preamble", "chunk_index": len(chunks)},
            ))

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        chunk_text = f"{header}\n\n{content}".strip()
        if chunk_text:
            section = re.sub(r"^#{1,6}\s+", "", header).strip()
            chunks.append(Chunk(
                text=chunk_text,
                metadata={**metadata, "strategy": "structure", "section": section, "chunk_index": len(chunks)},
            ))

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
