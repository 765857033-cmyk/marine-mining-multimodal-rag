from __future__ import annotations

import base64
import os
import re
from hashlib import blake2b
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import DocumentChunk
from .text_processing import normalize_text


FIGURE_CAPTION_PATTERN = re.compile(
    r"(?im)^\s*((fig\.?|figure|图|图版|plate|table|表)\s*[\dIVXivx\-\.]+[^\n]{0,260})"
)


def extract_multimodal_chunks(pdf_path: Path, config: AppConfig) -> list[DocumentChunk]:
    if not config.multimodal_enabled:
        return []
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install -r requirements.txt") from exc

    artifact_dir = config.upload_dir / "_multimodal" / pdf_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[DocumentChunk] = []
    image_count = 0

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_text = normalize_text(page.get_text("text", sort=True))
            captions = extract_captions(page_text)

            if config.extract_tables:
                chunks.extend(extract_table_chunks(page, pdf_path.name, page_index))

            if config.extract_images and image_count < config.max_images_per_pdf:
                image_chunks, used = extract_image_chunks(
                    doc=doc,
                    page=page,
                    source=pdf_path.name,
                    page_index=page_index,
                    page_text=page_text,
                    captions=captions,
                    artifact_dir=artifact_dir,
                    start_index=image_count,
                    max_count=config.max_images_per_pdf - image_count,
                    config=config,
                )
                chunks.extend(image_chunks)
                image_count += used

            if config.render_page_snapshots:
                snapshot = render_page_snapshot(
                    page=page,
                    source=pdf_path.name,
                    page_index=page_index,
                    page_text=page_text,
                    captions=captions,
                    artifact_dir=artifact_dir,
                    config=config,
                )
                if snapshot is not None:
                    chunks.append(snapshot)

    return chunks


def extract_table_chunks(page: Any, source: str, page_index: int) -> list[DocumentChunk]:
    try:
        tables = page.find_tables()
    except Exception:
        return []

    chunks: list[DocumentChunk] = []
    for table_index, table in enumerate(getattr(tables, "tables", []), start=1):
        try:
            rows = table.extract()
        except Exception:
            continue
        markdown = rows_to_markdown(rows)
        if len(markdown) < 40:
            continue
        text = (
            f"表格证据 | source={source} | page={page_index} | table={table_index}\n"
            f"该表格来自海洋矿产科研 PDF，可用于回答元素含量、样品指标、站位、矿物组成、"
            f"地球化学参数和对比分析问题。\n\n{markdown}"
        )
        chunks.append(
            DocumentChunk(
                chunk_id=make_multimodal_id(source, page_index, "table", table_index, markdown),
                source=source,
                page=page_index,
                text=text,
                modality="table",
                metadata={"table_index": table_index, "heading": f"Table {table_index}", "asset_path": ""},
            )
        )
    return chunks


def extract_image_chunks(
    doc: Any,
    page: Any,
    source: str,
    page_index: int,
    page_text: str,
    captions: list[str],
    artifact_dir: Path,
    start_index: int,
    max_count: int,
    config: AppConfig,
) -> tuple[list[DocumentChunk], int]:
    chunks: list[DocumentChunk] = []
    images = page.get_images(full=True)
    used = 0
    seen_xrefs: set[int] = set()
    for image_index, image_info in enumerate(images, start=1):
        if used >= max_count:
            break
        xref = int(image_info[0])
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            extracted = doc.extract_image(xref)
        except Exception:
            continue

        image_bytes = extracted.get("image", b"")
        ext = extracted.get("ext", "png")
        width = int(extracted.get("width", 0) or 0)
        height = int(extracted.get("height", 0) or 0)
        if len(image_bytes) < 2_000 or width < 80 or height < 80:
            continue

        asset_name = f"p{page_index:03d}_img{start_index + used + 1:03d}.{ext}"
        asset_path = artifact_dir / asset_name
        asset_path.write_bytes(image_bytes)
        caption = nearest_caption(captions, image_index)
        summary = summarize_image(
            image_path=asset_path,
            source=source,
            page=page_index,
            modality="image",
            caption=caption,
            page_text=page_text,
            config=config,
        )
        text = build_visual_chunk_text(
            source=source,
            page=page_index,
            modality="image",
            caption=caption,
            summary=summary,
            width=width,
            height=height,
            asset_path=asset_path,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=make_multimodal_id(source, page_index, "image", image_index, text),
                source=source,
                page=page_index,
                text=text,
                modality="image",
                metadata={
                    "image_index": image_index,
                    "asset_path": str(asset_path),
                    "caption": caption,
                    "width": width,
                    "height": height,
                    "heading": caption[:120] if caption else f"Image {image_index}",
                },
            )
        )
        used += 1
    return chunks, used


def render_page_snapshot(
    page: Any,
    source: str,
    page_index: int,
    page_text: str,
    captions: list[str],
    artifact_dir: Path,
    config: AppConfig,
) -> DocumentChunk | None:
    try:
        import fitz

        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        asset_path = artifact_dir / f"p{page_index:03d}_snapshot.png"
        pixmap.save(asset_path)
    except Exception:
        return None

    caption = "；".join(captions[:3])
    summary = summarize_image(
        image_path=asset_path,
        source=source,
        page=page_index,
        modality="page_snapshot",
        caption=caption,
        page_text=page_text,
        config=config,
    )
    text = build_visual_chunk_text(
        source=source,
        page=page_index,
        modality="page_snapshot",
        caption=caption,
        summary=summary,
        width=getattr(pixmap, "width", 0),
        height=getattr(pixmap, "height", 0),
        asset_path=asset_path,
    )
    return DocumentChunk(
        chunk_id=make_multimodal_id(source, page_index, "page_snapshot", 0, text),
        source=source,
        page=page_index,
        text=text,
        modality="page_snapshot",
        metadata={
            "asset_path": str(asset_path),
            "caption": caption,
            "width": getattr(pixmap, "width", 0),
            "height": getattr(pixmap, "height", 0),
            "heading": f"Page snapshot {page_index}",
        },
    )


def summarize_image(
    image_path: Path,
    source: str,
    page: int,
    modality: str,
    caption: str,
    page_text: str,
    config: AppConfig,
) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_visual_summary(source, page, modality, caption, page_text, image_path)
    try:
        from openai import OpenAI

        client = OpenAI()
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        prompt = (
            "你是海洋矿产科研论文的多模态解析助手。请阅读图片或页面截图，"
            "提取对问答检索有价值的信息：图/表类型、坐标轴或图例含义、样品或站位、"
            "元素/矿物/地球化学指标、趋势、结论和任何可引用证据。"
            "如果图片只是装饰或无法判断，请明确说明。"
            f"\nsource={source}, page={page}, modality={modality}"
            f"\n附近 caption：{caption or '无'}"
            f"\n页面文本摘录：{page_text[:1200]}"
        )
        response = client.chat.completions.create(
            model=config.vision_model,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
        )
        content = response.choices[0].message.content or ""
        return content[: config.max_image_summary_chars].strip()
    except Exception as exc:
        return fallback_visual_summary(source, page, modality, caption, page_text, image_path, error=str(exc))


def fallback_visual_summary(
    source: str,
    page: int,
    modality: str,
    caption: str,
    page_text: str,
    image_path: Path,
    error: str = "",
) -> str:
    caption_text = caption or "未识别到明确图题或表题"
    surrounding = extract_visual_context(page_text)
    error_note = f"；视觉模型未启用或调用失败：{error}" if error else "；未启用视觉模型"
    return (
        f"多模态视觉证据来自 {source} 第 {page} 页，类型为 {modality}，图片文件为 {image_path.name}"
        f"{error_note}。可检索信息主要来自图题/表题和页面邻近文本：{caption_text}。"
        f"页面相关上下文：{surrounding}"
    )


def build_visual_chunk_text(
    source: str,
    page: int,
    modality: str,
    caption: str,
    summary: str,
    width: int,
    height: int,
    asset_path: Path,
) -> str:
    return (
        f"多模态证据 | modality={modality} | source={source} | page={page}\n"
        f"caption: {caption or '无明确 caption'}\n"
        f"image_size: {width}x{height}\n"
        f"asset_path: {asset_path}\n\n"
        f"视觉/版面摘要：{summary}"
    )


def rows_to_markdown(rows: list[list[Any]]) -> str:
    normalized = [["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row] for row in rows]
    normalized = [row for row in normalized if any(cell for cell in row)]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    body = normalized[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_captions(page_text: str) -> list[str]:
    captions = [normalize_text(match.group(1)) for match in FIGURE_CAPTION_PATTERN.finditer(page_text)]
    return [caption for caption in captions if 4 <= len(caption) <= 280]


def nearest_caption(captions: list[str], image_index: int) -> str:
    if not captions:
        return ""
    return captions[min(image_index - 1, len(captions) - 1)]


def extract_visual_context(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    selected = [
        line
        for line in lines
        if re.search(r"(fig\.?|figure|图|图版|plate|table|表|sample|station|element|oxide|REE|稀土|元素|样品|站位)", line, re.I)
    ]
    if not selected:
        selected = lines[:6]
    return normalize_text(" ".join(selected[:8]))[:900]


def make_multimodal_id(source: str, page: int, modality: str, index: int, text: str) -> str:
    digest = blake2b(f"{source}:{page}:{modality}:{index}:{text[:100]}".encode("utf-8"), digest_size=8).hexdigest()
    return f"{source}:{page}:{modality}:{index}:{digest}"
