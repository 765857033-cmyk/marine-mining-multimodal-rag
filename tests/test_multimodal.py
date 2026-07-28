import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig
from src.models import DocumentChunk
from src.multimodal import build_visual_chunk_text, extract_captions, rows_to_markdown
from src.vector_store import EmbeddingProvider, VectorIndex


class MultimodalTests(unittest.TestCase):
    def test_rows_to_markdown_builds_table_evidence(self):
        markdown = rows_to_markdown([["Sample", "Co"], ["A1", "0.42"], ["A2", "0.51"]])
        self.assertIn("| Sample | Co |", markdown)
        self.assertIn("| A1 | 0.42 |", markdown)

    def test_extract_captions_finds_figure_and_table_titles(self):
        captions = extract_captions("Fig. 3 Distribution of polymetallic nodules\n正文\n表 2 元素含量")
        self.assertEqual(len(captions), 2)
        self.assertIn("polymetallic", captions[0])

    def test_hybrid_search_can_retrieve_visual_chunk(self):
        visual_text = build_visual_chunk_text(
            source="demo.pdf",
            page=5,
            modality="image",
            caption="Fig. 4 REE pattern of cobalt-rich crust",
            summary="该图展示富钴结壳稀土元素配分模式和 Ce 异常。",
            width=1200,
            height=800,
            asset_path=Path("figure.png"),
        )
        chunks = [
            DocumentChunk("text-1", "demo.pdf", 1, "多金属结核成矿环境。"),
            DocumentChunk("image-1", "demo.pdf", 5, visual_text, modality="image"),
        ]
        index = VectorIndex(EmbeddingProvider("hashing"), backend="numpy")
        index.build(chunks)
        results = index.search("富钴结壳 稀土元素 Ce 异常 图", top_k=2, keyword_top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.modality, "image")

    def test_load_old_index_without_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = VectorIndex(EmbeddingProvider("hashing"), backend="numpy")
            index.build([DocumentChunk("old", "old.pdf", 1, "海洋矿产文本证据")])
            index.save(Path(tmp))

            chunks_path = Path(tmp) / "chunks.json"
            payload = chunks_path.read_text(encoding="utf-8")
            payload = payload.replace(',\n      "modality": "text"', "")
            chunks_path.write_text(payload, encoding="utf-8")

            loaded = VectorIndex(EmbeddingProvider("hashing"), backend="numpy")
            self.assertTrue(loaded.load(Path(tmp)))
            self.assertEqual(loaded.search_chunks[0].modality, "text")
            self.assertEqual(loaded.parent_chunks[0].modality, "text")


if __name__ == "__main__":
    unittest.main()
