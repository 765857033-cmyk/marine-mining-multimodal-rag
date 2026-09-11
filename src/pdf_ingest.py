from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .mineru_ingest import ingest_pdf_with_mineru
from .models import DocumentChunk


def ingest_pdf(pdf_path: Path, config: AppConfig) -> list[DocumentChunk]:
    if config.parser_backend != "mineru":
        raise ValueError("项目已取消 PyMuPDF 兜底解析，请使用 PARSER_BACKEND=mineru。")
    return ingest_pdf_with_mineru(pdf_path, config)


def ingest_pdfs(pdf_paths: list[Path], config: AppConfig) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for pdf_path in pdf_paths:
        chunks.extend(ingest_pdf(pdf_path, config))
    return chunks
