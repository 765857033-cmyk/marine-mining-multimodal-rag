import sys
import types
import unittest

from src.models import DocumentChunk, RetrievalResult
from src.rerank import Reranker


class RerankBackendTests(unittest.TestCase):
    def test_rule_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            Reranker(backend="rule")

    def test_none_backend_keeps_retrieval_order_without_rule_scoring(self):
        reranker = Reranker(backend="none")
        results = [
            RetrievalResult(
                DocumentChunk("a", "demo.pdf", 1, "多金属结核 氧化还原 金属来源"),
                score=0.2,
                keyword_score=2.0,
            )
        ]
        ranked = reranker.rerank("多金属结核受什么控制", results, top_k=1)
        self.assertEqual(reranker.active_backend, "none")
        self.assertTrue(ranked)
        self.assertIsNone(ranked[0].rerank_score)
        self.assertIn("rerank=none", ranked[0].reason)

    def test_bge_backend_uses_flag_reranker_when_available(self):
        original = sys.modules.get("FlagEmbedding")
        fake_module = types.ModuleType("FlagEmbedding")

        class FakeFlagReranker:
            def __init__(self, model_name, use_fp16=False):
                self.model_name = model_name
                self.use_fp16 = use_fp16

            def compute_score(self, pairs):
                return [0.88 for _ in pairs]

        fake_module.FlagReranker = FakeFlagReranker
        sys.modules["FlagEmbedding"] = fake_module
        try:
            reranker = Reranker("BAAI/bge-reranker-v2-m3", backend="bge")
            results = [RetrievalResult(DocumentChunk("a", "demo.pdf", 1, "text"), score=0.1)]
            ranked = reranker.rerank("query", results, top_k=1)
            self.assertEqual(reranker.active_backend, "bge")
            self.assertEqual(ranked[0].rerank_score, 0.88)
            self.assertIn("rerank=bge", ranked[0].reason)
        finally:
            if original is None:
                sys.modules.pop("FlagEmbedding", None)
            else:
                sys.modules["FlagEmbedding"] = original


if __name__ == "__main__":
    unittest.main()
