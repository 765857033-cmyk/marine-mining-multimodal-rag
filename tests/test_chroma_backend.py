import tempfile
import unittest
from pathlib import Path

from src.models import DocumentChunk
from src.vector_store import EmbeddingProvider, VectorIndex


class ChromaBackendTests(unittest.TestCase):
    def test_chroma_backend_searches_and_roundtrips_with_numpy_fallback(self):
        index = VectorIndex(EmbeddingProvider("hashing"), backend="chroma")
        index.build(
            [
                DocumentChunk(
                    chunk_id="chroma-demo",
                    source="demo.pdf",
                    page=1,
                    text="polymetallic nodules are controlled by oxic conditions and sedimentation rate",
                )
            ],
            parent_child_enabled=False,
        )
        results = index.search("polymetallic nodules sedimentation", top_k=1, keyword_top_k=1)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "demo.pdf")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            index.save(Path(tmp))
            loaded = VectorIndex(EmbeddingProvider("hashing"), backend="chroma")
            self.assertTrue(loaded.load(Path(tmp)))
            loaded_results = loaded.search("polymetallic nodules sedimentation", top_k=1, keyword_top_k=1)
            self.assertTrue(loaded_results)
            self.assertEqual(loaded_results[0].chunk.source, "demo.pdf")


if __name__ == "__main__":
    unittest.main()
