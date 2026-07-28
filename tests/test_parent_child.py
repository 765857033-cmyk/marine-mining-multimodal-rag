import tempfile
import unittest
from pathlib import Path

from src.models import DocumentChunk
from src.parent_child import build_parent_child_chunks
from src.vector_store import EmbeddingProvider, VectorIndex


class ParentChildTests(unittest.TestCase):
    def test_build_parent_child_chunks(self):
        parent_text = "多金属结核成矿环境包括氧化还原条件、沉积速率和金属来源。" * 20
        parents, children = build_parent_child_chunks(
            [DocumentChunk("raw-1", "demo.pdf", 1, parent_text)],
            child_chunk_size=180,
            child_chunk_overlap=30,
        )
        self.assertEqual(len(parents), 1)
        self.assertGreater(len(children), 1)
        self.assertEqual(children[0].parent_id, parents[0].parent_id)
        self.assertEqual(children[0].chunk_role, "child")

    def test_search_promotes_child_hit_to_parent(self):
        parent_text = (
            "第一部分介绍背景。" * 20
            + " 稀有标记词 xyz-parent-child-marker 表明这里是关键证据。"
            + " 后续部分给出完整上下文。" * 20
        )
        index = VectorIndex(EmbeddingProvider("hashing"), backend="numpy")
        index.build(
            [DocumentChunk("raw-1", "demo.pdf", 2, parent_text)],
            parent_child_enabled=True,
            child_chunk_size=120,
            child_chunk_overlap=20,
        )
        results = index.search("xyz-parent-child-marker", top_k=3, keyword_top_k=3)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.chunk_role, "parent")
        self.assertIn("完整上下文", results[0].chunk.text)
        self.assertIn("matched_child_text", results[0].chunk.metadata)

    def test_parent_child_index_roundtrip(self):
        index = VectorIndex(EmbeddingProvider("hashing"), backend="numpy")
        index.build(
            [DocumentChunk("raw-1", "demo.pdf", 1, "多金属结核成矿环境。" * 40)],
            parent_child_enabled=True,
            child_chunk_size=160,
            child_chunk_overlap=20,
        )
        with tempfile.TemporaryDirectory() as tmp:
            index.save(Path(tmp))
            loaded = VectorIndex(EmbeddingProvider("hashing"), backend="numpy")
            self.assertTrue(loaded.load(Path(tmp)))
            self.assertTrue(loaded.parent_chunks)
            self.assertTrue(loaded.search_chunks)
            self.assertGreaterEqual(len(loaded.search_chunks), len(loaded.parent_chunks))


if __name__ == "__main__":
    unittest.main()
