import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import AppConfig
from src.mineru_ingest import mineru_content_list_chunks, mineru_markdown_chunks
from src.models import DocumentChunk
from src.pdf_ingest import ingest_pdf


class MinerUBackendTests(unittest.TestCase):
    def test_content_list_creates_text_and_table_chunks(self):
        items = [
            {"type": "text", "page_idx": 0, "text": "polymetallic nodules form under oxic conditions"},
            {"type": "table", "page_idx": 1, "table_caption": "Co content", "table_body": "| Sample | Co |\n| A1 | 0.42 |"},
        ]
        chunks = mineru_content_list_chunks(items, "demo.pdf", AppConfig(), Path("."))
        self.assertTrue(chunks)
        self.assertTrue(any(chunk.metadata["parser"] == "mineru" for chunk in chunks))
        self.assertTrue(any(chunk.modality == "table" for chunk in chunks))

    def test_markdown_fallback_creates_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            markdown = Path(tmp) / "demo.md"
            markdown.write_text("海洋多金属结核受氧化还原条件、沉积速率和金属来源控制。" * 10, encoding="utf-8")
            chunks = mineru_markdown_chunks(Path(tmp), "demo.pdf", AppConfig(chunk_size=300, chunk_overlap=40))
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].metadata["parser"], "mineru")

    def test_ingest_pdf_prefers_mineru_backend(self):
        expected = [
            DocumentChunk(
                chunk_id="mineru-1",
                source="demo.pdf",
                page=1,
                text="MinerU chunk",
                modality="text",
                metadata={"parser": "mineru"},
            )
        ]
        with patch("src.pdf_ingest.ingest_pdf_with_mineru", return_value=expected) as mocked:
            chunks = ingest_pdf(Path("demo.pdf"), AppConfig(parser_backend="mineru", multimodal_enabled=False))
        mocked.assert_called_once()
        self.assertEqual(chunks, expected)


if __name__ == "__main__":
    unittest.main()
