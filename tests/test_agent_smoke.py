import tempfile
import unittest
from pathlib import Path

from src.agent import MiningRagAgent, rewrite_queries
from src.config import AppConfig
from src.models import DocumentChunk
from src.vector_store import EmbeddingProvider, VectorIndex


class AgentSmokeTests(unittest.TestCase):
    def test_agent_returns_sources_for_domain_question(self):
        chunks = [
            DocumentChunk(
                chunk_id="demo-1",
                source="demo.pdf",
                page=3,
                text="多金属结核成矿环境通常与氧化还原界面、低沉积速率、金属来源和海底地形有关。",
            ),
            DocumentChunk(
                chunk_id="demo-2",
                source="demo.pdf",
                page=8,
                text="富钴结壳多发育于海山硬质基底，其元素富集受水深、底流和磷酸盐化影响。",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                top_k=2,
                keyword_top_k=2,
                rerank_top_k=2,
                state_db_path=Path(tmp) / "state.db",
                trace_dir=Path(tmp) / "traces",
            )
            index = VectorIndex(EmbeddingProvider("hashing"), backend="chroma")
            index.build(chunks)
            answer = MiningRagAgent(index, config).invoke(
                "多金属结核成矿受哪些环境因素控制？",
                session_id="smoke-session",
            )
            self.assertEqual(answer.route, "rag")
            self.assertTrue(answer.sources)
            self.assertIn("demo.pdf", answer.answer)
            self.assertGreaterEqual(answer.retrieval_rounds, 1)
            self.assertTrue(any("hybrid_retrieval" in step for step in answer.trace))
            self.assertTrue(any("source_verification" in step for step in answer.trace))
            self.assertTrue(any(event["node"] == "citation_validation" for event in answer.trace_events))
            self.assertTrue(answer.citation_validation["passed"])
            self.assertTrue((config.trace_dir / f"{answer.trace_id}.json").exists())
            self.assertGreaterEqual(len(answer.query_variants), 2)

    def test_hybrid_retrieval_includes_keyword_signal(self):
        chunks = [
            DocumentChunk(
                chunk_id="rare-keyword",
                source="keyword.pdf",
                page=1,
                text="Abyssal cobalt flux marker xyz-special-token appears in this paragraph.",
            ),
            DocumentChunk(
                chunk_id="generic",
                source="generic.pdf",
                page=2,
                text="This paragraph discusses marine minerals and deep sea resource assessment.",
            ),
        ]
        index = VectorIndex(EmbeddingProvider("hashing"), backend="chroma")
        index.build(chunks)
        results = index.search("xyz-special-token", top_k=1, keyword_top_k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.metadata["retrieved_parent_id"], results[0].chunk.parent_id)
        self.assertIn("xyz-special-token", results[0].chunk.metadata["matched_child_text"])
        self.assertGreater(results[0].keyword_score, 0)
        self.assertIn("bm25_rank", results[0].reason)

    def test_query_rewrite_generates_multiple_variants(self):
        variants = rewrite_queries("多金属结核成矿受哪些环境因素控制？", max_queries=4)
        self.assertGreaterEqual(len(variants), 3)
        self.assertTrue(any("polymetallic nodules" in query for query in variants))
        self.assertTrue(any("controlling factors" in query for query in variants))


if __name__ == "__main__":
    unittest.main()
