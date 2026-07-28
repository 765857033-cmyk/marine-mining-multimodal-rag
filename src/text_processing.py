from __future__ import annotations

import re
from collections import Counter
from hashlib import blake2b
from typing import Iterable

from .models import DocumentChunk


DOMAIN_TERMS = {
    "海洋矿产",
    "深海",
    "多金属结核",
    "富钴结壳",
    "热液硫化物",
    "稀土",
    "镍",
    "钴",
    "锰",
    "铜",
    "赋存环境",
    "控制因素",
    "识别指标",
    "沉积物",
    "成矿",
    "矿物学",
    "地球化学",
    "polymetallic nodules",
    "cobalt-rich crust",
    "seafloor massive sulfide",
    "hydrothermal sulfide",
    "ferromanganese",
    "rare earth elements",
}

CASUAL_PATTERNS = [
    r"^(你好|您好|hello|hi|嗨)\b",
    r"(你是谁|能做什么|怎么用)",
    r"(谢谢|感谢|thanks)",
]


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"([A-Za-z])-[\n\r]+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_fingerprint(line: str) -> str:
    cleaned = re.sub(r"\d+", "#", line.lower())
    cleaned = re.sub(r"\W+", "", cleaned)
    return cleaned[:80]


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if re.fullmatch(r"[-_ ]*\d+[-_ ]*", s):
        return True
    if re.fullmatch(r"(page|第)\s*\d+(\s*/\s*\d+)?", s, re.I):
        return True
    if re.search(r"(downloaded from|copyright|all rights reserved|www\.|http://|https://)", s, re.I):
        return True
    if len(s) < 4 and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s):
        return True
    return False


def clean_pdf_pages(pages: Iterable[tuple[int, str]]) -> list[tuple[int, str]]:
    raw_pages = [(page, normalize_text(text)) for page, text in pages]
    fingerprints: Counter[str] = Counter()
    for _, text in raw_pages:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:4] + lines[-4:]:
            fp = _line_fingerprint(line)
            if len(fp) >= 8:
                fingerprints[fp] += 1

    repeated = {fp for fp, count in fingerprints.items() if count >= 3}
    cleaned_pages: list[tuple[int, str]] = []
    for page, text in raw_pages:
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            fp = _line_fingerprint(stripped)
            if _is_noise_line(stripped) or fp in repeated:
                continue
            kept.append(stripped)
        cleaned_pages.append((page, normalize_text("\n".join(kept))))
    return cleaned_pages


def strip_reference_tail(text: str) -> str:
    match = re.search(r"\n\s*(references|参考文献)\s*\n", text, re.I)
    if match and match.start() > len(text) * 0.55:
        return text[: match.start()].strip()
    return text


def extract_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if 6 <= len(line) <= 120 and re.match(r"^(\d+(\.\d+)*\.?\s+)?[A-Z\u4e00-\u9fff]", line):
            return line[:120]
    return ""


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += max(1, chunk_size - overlap)
            current = ""
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    with_overlap: list[str] = []
    previous_tail = ""
    for chunk in chunks:
        merged = f"{previous_tail}\n{chunk}".strip() if previous_tail else chunk
        with_overlap.append(merged[: chunk_size + overlap])
        previous_tail = chunk[-overlap:]
    return with_overlap


def make_chunk_id(source: str, page: int, index: int, text: str) -> str:
    digest = blake2b(f"{source}:{page}:{index}:{text[:100]}".encode("utf-8"), digest_size=8).hexdigest()
    return f"{source}:{page}:{index}:{digest}"


def pages_to_chunks(
    pages: Iterable[tuple[int, str]],
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    cleaned_pages = clean_pdf_pages(pages)
    all_text = strip_reference_tail("\n\n".join(text for _, text in cleaned_pages))
    page_lookup = {text: page for page, text in cleaned_pages}
    chunks: list[DocumentChunk] = []
    running_index = 0
    for page, page_text in cleaned_pages:
        if page_text and page_text not in all_text:
            continue
        heading = extract_heading(page_text)
        for local_index, chunk_text in enumerate(split_text(page_text, chunk_size, overlap)):
            if len(chunk_text) < 80:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source, page, local_index, chunk_text),
                    source=source,
                    page=page_lookup.get(page_text, page),
                    text=chunk_text,
                    modality="text",
                    metadata={"heading": heading, "index": running_index},
                )
            )
            running_index += 1
    return chunks


def is_casual_question(question: str) -> bool:
    q = question.strip().lower()
    return any(re.search(pattern, q, re.I) for pattern in CASUAL_PATTERNS)


def domain_term_hits(text: str) -> int:
    lowered = text.lower()
    return sum(1 for term in DOMAIN_TERMS if term.lower() in lowered)
