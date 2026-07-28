from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .docling_ingest import DoclingUnavailableError, ingest_pdf_with_docling
from .mineru_ingest import MinerUUnavailableError, ingest_pdf_with_mineru
from .models import DocumentChunk
from .multimodal import extract_multimodal_chunks
from .text_processing import pages_to_chunks


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install -r requirements.txt") from exc

    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            pages.append((page_index, text))
    return pages


def ingest_pdf(pdf_path: Path, config: AppConfig) -> list[DocumentChunk]:
    chunk_size = config.parent_chunk_size if config.parent_child_enabled else config.chunk_size
    chunk_overlap = config.parent_chunk_overlap if config.parent_child_enabled else config.chunk_overlap
    if config.parser_backend in {"mineru", "auto"}:
        try:
            chunks = ingest_pdf_with_mineru(pdf_path, config)
            if chunks:
                return maybe_add_multimodal_chunks(pdf_path, chunks, config)
        except MinerUUnavailableError:
            if config.parser_backend == "mineru" and not config.parser_fallback:
                raise
        except Exception:
            if config.parser_backend == "mineru" and not config.parser_fallback:
                raise
        if config.parser_backend == "mineru" and not config.parser_fallback:
            return []

    if config.parser_backend in {"docling", "auto"}:
        try:
            chunks = ingest_pdf_with_docling(pdf_path, config)
            if chunks:
                return maybe_add_multimodal_chunks(pdf_path, chunks, config)
        except DoclingUnavailableError:
            if config.parser_backend == "docling" and not config.parser_fallback:
                raise
        except Exception:
            if config.parser_backend == "docling" and not config.parser_fallback:
                raise
        if config.parser_backend == "docling" and not config.parser_fallback:
            return []

    pages = extract_pdf_pages(pdf_path)
    text_chunks = pages_to_chunks(
        pages,
        source=pdf_path.name,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    multimodal_chunks = extract_multimodal_chunks(pdf_path, config)
    return text_chunks + multimodal_chunks


def maybe_add_multimodal_chunks(pdf_path: Path, chunks: list[DocumentChunk], config: AppConfig) -> list[DocumentChunk]:
    if config.multimodal_enabled and pdf_path.exists() and (config.extract_images or config.render_page_snapshots):
        try:
            return chunks + extract_multimodal_chunks(pdf_path, config)
        except Exception:
            return chunks
    return chunks


def ingest_pdfs(pdf_paths: list[Path], config: AppConfig) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for pdf_path in pdf_paths:
        chunks.extend(ingest_pdf(pdf_path, config))
    return chunks
