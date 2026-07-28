import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import AppConfig
from src.docling_ingest import docling_markdown_chunks
from src.models import DocumentChunk
from src.pdf_ingest import ingest_pdf


class FakeDoclingDocument:
    def export_to_markdown(self):
        return "多金属结核成矿环境包括氧化还原条件、沉积速率和金属来源。" * 10


class DoclingBackendTests(unittest.TestCase):
    def test_docling_markdown_fallback_creates_chunks(self):
        chunks = docling_markdown_chunks(
            FakeDoclingDocument(),
            source="demo.pdf",
            config=AppConfig(parser_backend="docling", chunk_size=300, chunk_overlap=40),
        )
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].metadata["parser"], "docling")

    def test_ingest_pdf_prefers_docling_backend(self):
        expected = [
            DocumentChunk(
                chunk_id="docling-1",
                source="demo.pdf",
                page=1,
                text="Docling chunk",
                modality="text",
                metadata={"parser": "docling"},
            )
        ]
        with patch("src.pdf_ingest.ingest_pdf_with_docling", return_value=expected) as mocked:
            chunks = ingest_pdf(Path("demo.pdf"), AppConfig(parser_backend="docling"))
        mocked.assert_called_once()
        self.assertEqual(chunks, expected)


if __name__ == "__main__":
    unittest.main()
