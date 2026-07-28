from __future__ import annotations

import json
import re
from dataclasses import asdict
from hashlib import blake2b
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import DocumentChunk
from .multimodal import rows_to_markdown
from .text_processing import normalize_text, pages_to_chunks


class DoclingUnavailableError(RuntimeError):
    pass


def ingest_pdf_with_docling(pdf_path: Path, config: AppConfig) -> list[DocumentChunk]:
    result = convert_with_docling(pdf_path, config)
    doc = getattr(result, "document", result)
    artifact_dir = config.upload_dir / "_docling" / pdf_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if config.docling_export_artifacts:
        export_docling_artifacts(doc, artifact_dir)

    chunks: list[DocumentChunk] = []
    if config.docling_use_hybrid_chunker:
        chunks.extend(docling_hybrid_chunks(doc, pdf_path.name, config))

    if not chunks:
        chunks.extend(docling_markdown_chunks(doc, pdf_path.name, config))

    if config.docling_include_tables:
        chunks.extend(docling_table_chunks(doc, pdf_path.name))

    return dedupe_chunks(chunks)


def convert_with_docling(pdf_path: Path, config: AppConfig) -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as exc:
        raise DoclingUnavailableError(
            "Docling is not installed. Run install_docling.ps1 or pip install docling langchain-docling."
        ) from exc

    pipeline_options = PdfPipelineOptions()
    set_if_present(pipeline_options, "do_ocr", config.docling_do_ocr)
    set_if_present(pipeline_options, "do_table_structure", config.docling_do_table_structure)
    set_if_present(pipeline_options, "generate_page_images", config.render_page_snapshots)
    set_if_present(pipeline_options, "generate_picture_images", config.extract_images)

    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    except Exception:
        converter = DocumentConverter()

    return converter.convert(str(pdf_path))


def set_if_present(obj: Any, attr: str, value: Any) -> None:
    if hasattr(obj, attr):
        try:
            setattr(obj, attr, value)
        except Exception:
            pass


def docling_hybrid_chunks(doc: Any, source: str, config: AppConfig) -> list[DocumentChunk]:
    try:
        from docling.chunking import HybridChunker
    except Exception:
        return []

    try:
        target_size = config.parent_chunk_size if config.parent_child_enabled else config.chunk_size
        chunker = HybridChunker(max_tokens=max(128, target_size))
    except Exception:
        chunker = HybridChunker()

    chunks: list[DocumentChunk] = []
    try:
        raw_chunks = list(chunker.chunk(dl_doc=doc))
    except TypeError:
        try:
            raw_chunks = list(chunker.chunk(doc))
        except Exception:
            return []
    except Exception:
        return []

    for index, raw_chunk in enumerate(raw_chunks):
        text = contextualize_chunk(chunker, raw_chunk)
        if len(text) < 40:
            continue
        metadata = extract_docling_metadata(raw_chunk)
        page = metadata.get("page", 1)
        modality = infer_modality(text, metadata)
        chunks.append(
            DocumentChunk(
                chunk_id=make_docling_id(source, page, modality, index, text),
                source=source,
                page=page,
                text=text,
                modality=modality,
                metadata={
                    **metadata,
                    "parser": "docling",
                    "heading": metadata.get("heading", ""),
                    "index": index,
                },
            )
        )
    return chunks


def docling_markdown_chunks(doc: Any, source: str, config: AppConfig) -> list[DocumentChunk]:
    markdown = export_to_markdown(doc)
    if not markdown:
        return []
    pages = split_markdown_into_pseudo_pages(markdown)
    chunk_size = config.parent_chunk_size if config.parent_child_enabled else config.chunk_size
    overlap = config.parent_chunk_overlap if config.parent_child_enabled else config.chunk_overlap
    chunks = pages_to_chunks(pages, source=source, chunk_size=chunk_size, overlap=overlap)
    for chunk in chunks:
        chunk.metadata["parser"] = "docling"
        chunk.metadata["docling_export"] = "markdown"
    return chunks


def docling_table_chunks(doc: Any, source: str) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    tables = getattr(doc, "tables", []) or []
    for table_index, table in enumerate(tables, start=1):
        markdown = table_to_markdown(table)
        if len(markdown) < 40:
            continue
        page = extract_page_from_item(table)
        heading = f"Docling table {table_index}"
        text = (
            f"Docling 表格证据 | source={source} | page={page} | table={table_index}\n"
            f"该表格由 Docling 结构化解析，可用于样品、站位、元素含量、矿物组成、"
            f"地球化学指标和对比分析。\n\n{markdown}"
        )
        chunks.append(
            DocumentChunk(
                chunk_id=make_docling_id(source, page, "table", table_index, text),
                source=source,
                page=page,
                text=text,
                modality="table",
                metadata={"parser": "docling", "table_index": table_index, "heading": heading},
            )
        )
    return chunks


def export_docling_artifacts(doc: Any, artifact_dir: Path) -> None:
    markdown = export_to_markdown(doc)
    if markdown:
        (artifact_dir / "docling.md").write_text(markdown, encoding="utf-8")
    json_payload = export_to_json(doc)
    if json_payload:
        (artifact_dir / "docling.json").write_text(json_payload, encoding="utf-8")


def contextualize_chunk(chunker: Any, raw_chunk: Any) -> str:
    for attr in ("contextualize",):
        method = getattr(chunker, attr, None)
        if callable(method):
            try:
                text = method(raw_chunk)
                if text:
                    return normalize_text(str(text))
            except Exception:
                pass

    for attr in ("text", "content"):
        value = getattr(raw_chunk, attr, "")
        if value:
            return normalize_text(str(value))

    if isinstance(raw_chunk, dict):
        for key in ("text", "content"):
            if raw_chunk.get(key):
                return normalize_text(str(raw_chunk[key]))
    return normalize_text(str(raw_chunk))


def extract_docling_metadata(raw_chunk: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    meta = getattr(raw_chunk, "meta", None) or getattr(raw_chunk, "metadata", None)
    if meta is not None:
        metadata.update(flatten_metadata(meta))
    if isinstance(raw_chunk, dict):
        metadata.update(flatten_metadata(raw_chunk.get("meta") or raw_chunk.get("metadata") or {}))

    page = metadata.get("page") or metadata.get("page_no") or metadata.get("page_number")
    if not page:
        page = first_int_from_values(metadata.values())
    metadata["page"] = safe_int(page, default=1)

    headings = metadata.get("headings") or metadata.get("heading") or metadata.get("section_header") or ""
    if isinstance(headings, list):
        headings = " > ".join(str(item) for item in headings if item)
    metadata["heading"] = str(headings)[:180]
    return metadata


def flatten_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = asdict(value)
        except Exception:
            raw = getattr(value, "__dict__", {}) or {}

    flattened: dict[str, Any] = {}
    for key, item in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            flattened[key] = item
        elif isinstance(item, list):
            flattened[key] = [simple_value(entry) for entry in item[:8]]
        elif isinstance(item, dict):
            for sub_key, sub_value in item.items():
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    flattened[f"{key}_{sub_key}"] = sub_value
    return flattened


def simple_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    for attr in ("text", "label", "name", "page_no", "page"):
        attr_value = getattr(value, attr, None)
        if attr_value:
            return attr_value
    return str(value)


def infer_modality(text: str, metadata: dict[str, Any]) -> str:
    label = " ".join(str(value).lower() for value in metadata.values())
    lowered = text.lower()
    if "table" in label or re.search(r"(^|\n)\|.+\|", text):
        return "table"
    if any(token in label for token in ("picture", "image", "figure")):
        return "image"
    if any(token in lowered for token in ("figure", "fig.", "图 ", "图版", "caption")):
        return "image"
    return "text"


def table_to_markdown(table: Any) -> str:
    for call in (
        lambda: table.export_to_dataframe().to_markdown(index=False),
        lambda: table.export_to_dataframe().to_csv(index=False),
        lambda: table.export_to_markdown(),
        lambda: table.export_to_html(),
    ):
        try:
            value = call()
            if value:
                return str(value)
        except Exception:
            pass

    data = getattr(table, "data", None)
    if data is not None:
        rows = getattr(data, "table_cells", None) or getattr(data, "grid", None)
        if rows:
            return rows_to_markdown(rows)
    return normalize_text(str(table))


def export_to_markdown(doc: Any) -> str:
    for method_name in ("export_to_markdown", "export_to_md"):
        method = getattr(doc, method_name, None)
        if callable(method):
            try:
                value = method()
                if value:
                    return normalize_text(str(value))
            except Exception:
                pass
    return ""


def export_to_json(doc: Any) -> str:
    for method_name in ("export_to_dict", "model_dump", "dict"):
        method = getattr(doc, method_name, None)
        if callable(method):
            try:
                value = method()
                return json.dumps(value, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
    return ""


def split_markdown_into_pseudo_pages(markdown: str) -> list[tuple[int, str]]:
    parts = re.split(r"\n\s*<!--\s*page[^>]*-->\s*\n", markdown, flags=re.I)
    if len(parts) <= 1:
        return [(1, markdown)]
    return [(index, part) for index, part in enumerate(parts, start=1) if part.strip()]


def extract_page_from_item(item: Any) -> int:
    metadata = flatten_metadata(getattr(item, "meta", None) or getattr(item, "metadata", None) or item)
    page = metadata.get("page") or metadata.get("page_no") or metadata.get("page_number")
    if not page:
        page = first_int_from_values(metadata.values())
    return safe_int(page, 1)


def first_int_from_values(values: Any) -> int | None:
    for value in values:
        text = str(value)
        match = re.search(r"\b([1-9]\d{0,3})\b", text)
        if match:
            return int(match.group(1))
    return None


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def make_docling_id(source: str, page: int, modality: str, index: int, text: str) -> str:
    digest = blake2b(f"docling:{source}:{page}:{modality}:{index}:{text[:120]}".encode("utf-8"), digest_size=8)
    return f"{source}:{page}:docling-{modality}:{index}:{digest.hexdigest()}"


def dedupe_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    seen: set[str] = set()
    deduped: list[DocumentChunk] = []
    for chunk in chunks:
        fingerprint = blake2b(chunk.text[:600].encode("utf-8"), digest_size=8).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(chunk)
    return deduped
