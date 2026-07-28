from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from hashlib import blake2b
from pathlib import Path
from typing import Any

from .config import AppConfig
from .models import DocumentChunk
from .text_processing import normalize_text, pages_to_chunks


class MinerUUnavailableError(RuntimeError):
    pass


def ingest_pdf_with_mineru(pdf_path: Path, config: AppConfig) -> list[DocumentChunk]:
    artifact_dir = config.upload_dir / "_mineru" / pdf_path.stem
    artifact_dir.mkdir(parents=True, exist_ok=True)

    run_mineru(pdf_path, artifact_dir, config)

    chunks: list[DocumentChunk] = []
    content_items = load_content_list_items(artifact_dir)
    if content_items:
        chunks.extend(mineru_content_list_chunks(content_items, pdf_path.name, config, artifact_dir))

    if not chunks:
        chunks.extend(mineru_markdown_chunks(artifact_dir, pdf_path.name, config))

    return dedupe_chunks(chunks)


def run_mineru(pdf_path: Path, artifact_dir: Path, config: AppConfig) -> None:
    command = shlex.split(config.mineru_command)
    if not command:
        raise MinerUUnavailableError("MINERU_COMMAND is empty.")
    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        raise MinerUUnavailableError(
            "MinerU is not installed or not on PATH. Install it with install_mineru.ps1, "
            "or set MINERU_COMMAND to your mineru executable."
        )

    cmd = command + ["-p", str(pdf_path), "-o", str(artifact_dir)]
    if config.mineru_backend:
        cmd.extend(["-b", config.mineru_backend])
    if config.mineru_method:
        cmd.extend(["-m", config.mineru_method])
    if config.mineru_extra_args:
        cmd.extend(shlex.split(config.mineru_extra_args))

    completed = subprocess.run(
        cmd,
        cwd=str(pdf_path.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.mineru_timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"MinerU parse failed with code {completed.returncode}: {message[-1200:]}")


def load_content_list_items(artifact_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(
        artifact_dir.rglob("*content_list*.json"),
        key=lambda path: (path.stat().st_mtime, len(path.parts)),
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = extract_items(payload)
        if items:
            return items
    return []


def extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("content_list", "items", "pages", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                if key == "pages":
                    return flatten_page_items(value)
                return [item for item in value if isinstance(item, dict)]
    return []


def flatten_page_items(pages: list[Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        page_items = page.get("items") or page.get("blocks") or page.get("content") or []
        if not isinstance(page_items, list):
            continue
        for item in page_items:
            if isinstance(item, dict):
                item.setdefault("page_idx", page.get("page_idx", page_index - 1))
                items.append(item)
    return items


def mineru_content_list_chunks(
    items: list[dict[str, Any]], source: str, config: AppConfig, artifact_dir: Path
) -> list[DocumentChunk]:
    page_lines: dict[int, list[str]] = {}
    element_chunks: list[DocumentChunk] = []

    for index, item in enumerate(items):
        page = extract_page(item)
        modality = infer_mineru_modality(item)
        text = item_to_text(item)
        if not text:
            continue

        prefix = {"table": "[Table]", "image": "[Figure]", "formula": "[Formula]"}.get(modality, "")
        page_lines.setdefault(page, []).append(f"{prefix} {text}".strip())

        if modality in {"table", "image", "formula"} and len(text) >= 20:
            asset_path = resolve_asset_path(item, artifact_dir)
            element_chunks.append(
                DocumentChunk(
                    chunk_id=make_mineru_id(source, page, modality, index, text),
                    source=source,
                    page=page,
                    text=build_element_text(source, page, modality, text),
                    modality=modality,
                    metadata={
                        "parser": "mineru",
                        "mineru_export": "content_list",
                        "index": index,
                        "asset_path": str(asset_path) if asset_path else "",
                        "heading": item_heading(item),
                    },
                )
            )

    chunk_size = config.parent_chunk_size if config.parent_child_enabled else config.chunk_size
    overlap = config.parent_chunk_overlap if config.parent_child_enabled else config.chunk_overlap
    pages = [(page, "\n".join(lines)) for page, lines in sorted(page_lines.items()) if lines]
    text_chunks = pages_to_chunks(pages, source=source, chunk_size=chunk_size, overlap=overlap)
    for chunk in text_chunks:
        chunk.metadata["parser"] = "mineru"
        chunk.metadata["mineru_export"] = "content_list"
    return text_chunks + element_chunks


def mineru_markdown_chunks(artifact_dir: Path, source: str, config: AppConfig) -> list[DocumentChunk]:
    markdown_parts: list[str] = []
    for markdown_path in sorted(artifact_dir.rglob("*.md")):
        try:
            text = normalize_text(markdown_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if text:
            markdown_parts.append(text)
    if not markdown_parts:
        return []

    chunk_size = config.parent_chunk_size if config.parent_child_enabled else config.chunk_size
    overlap = config.parent_chunk_overlap if config.parent_child_enabled else config.chunk_overlap
    pages = [(index, text) for index, text in enumerate(markdown_parts, start=1)]
    chunks = pages_to_chunks(pages, source=source, chunk_size=chunk_size, overlap=overlap)
    for chunk in chunks:
        chunk.metadata["parser"] = "mineru"
        chunk.metadata["mineru_export"] = "markdown"
    return chunks


def item_to_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "text",
        "content",
        "md_content",
        "table_body",
        "table_caption",
        "img_caption",
        "image_caption",
        "caption",
        "latex",
    ):
        value = item.get(key)
        if isinstance(value, list):
            parts.extend(str(entry) for entry in value if entry)
        elif value:
            parts.append(str(value))
    return normalize_text("\n".join(parts))


def infer_mineru_modality(item: dict[str, Any]) -> str:
    value = " ".join(str(item.get(key, "")).lower() for key in ("type", "category", "block_type"))
    if "table" in value:
        return "table"
    if any(token in value for token in ("image", "figure", "img")):
        return "image"
    if any(token in value for token in ("formula", "equation", "latex")):
        return "formula"
    if item.get("table_body") or item.get("table_caption"):
        return "table"
    if item.get("img_path") or item.get("image_path") or item.get("img_caption"):
        return "image"
    return "text"


def extract_page(item: dict[str, Any]) -> int:
    for key in ("page", "page_no", "page_number"):
        if key in item:
            return safe_int(item.get(key), 1)
    if "page_idx" in item:
        return safe_int(item.get("page_idx"), 0) + 1
    return 1


def resolve_asset_path(item: dict[str, Any], artifact_dir: Path) -> Path | None:
    for key in ("img_path", "image_path", "path"):
        value = item.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        for candidate in (artifact_dir / path, artifact_dir / path.name):
            if candidate.exists():
                return candidate
    return None


def item_heading(item: dict[str, Any]) -> str:
    for key in ("section", "heading", "title"):
        value = item.get(key)
        if value:
            return str(value)[:180]
    return ""


def build_element_text(source: str, page: int, modality: str, text: str) -> str:
    label = {"table": "表格", "image": "图片/图注", "formula": "公式"}.get(modality, modality)
    return (
        f"MinerU {label}证据 | source={source} | page={page}\n"
        f"该内容由 MinerU 从 PDF 版面中结构化解析，可用于科研文献问答、来源追溯和多模态证据检索。\n\n"
        f"{text}"
    )


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def make_mineru_id(source: str, page: int, modality: str, index: int, text: str) -> str:
    digest = blake2b(f"mineru:{source}:{page}:{modality}:{index}:{text[:120]}".encode("utf-8"), digest_size=8)
    return f"{source}:{page}:mineru-{modality}:{index}:{digest.hexdigest()}"


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
