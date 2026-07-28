import unittest

from src.models import AgentAnswer, DocumentChunk, RetrievalResult
from tools.evaluate import evaluate_answer, normalize_expected_sources


class EvaluateMetricTests(unittest.TestCase):
    def test_hit_mrr_and_citation_metrics(self):
        answer = AgentAnswer(
            answer="多金属结核受氧化还原条件控制 [2]",
            route="rag",
            need_retrieval=True,
            sources=[
                RetrievalResult(DocumentChunk("a", "other.pdf", 1, "无关内容"), score=0.2),
                RetrievalResult(DocumentChunk("b", "demo.pdf", 5, "氧化还原条件 控制 多金属结核"), score=0.9),
            ],
            rewritten_query="",
            verification_passed=True,
            retrieval_rounds=1,
        )
        metrics = evaluate_answer(
            {
                "question": "多金属结核受什么控制？",
                "expected_sources": [{"source": "demo.pdf", "page": 5}],
                "expected_terms": ["氧化还原", "多金属结核"],
                "min_term_recall": 0.5,
            },
            answer,
            hit_ks=(1, 3),
        )
        self.assertFalse(metrics["hit_at_1"])
        self.assertTrue(metrics["hit_at_3"])
        self.assertEqual(metrics["first_relevant_rank"], 2)
        self.assertAlmostEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["citation_accuracy"], 1.0)
        self.assertEqual(metrics["faithfulness"], 1.0)

    def test_refusal_accuracy(self):
        answer = AgentAnswer(
            answer="当前证据不足，无法可靠回答。",
            route="rag",
            need_retrieval=True,
            sources=[],
            rewritten_query="",
            verification_passed=False,
            retrieval_rounds=2,
        )
        metrics = evaluate_answer({"question": "不存在的问题", "should_refuse": True}, answer)
        self.assertTrue(metrics["refusal_correct"])
        self.assertTrue(metrics["passed"])

    def test_old_expected_source_string_is_supported(self):
        normalized = normalize_expected_sources(["demo.pdf"])
        self.assertEqual(normalized[0]["source"], "demo.pdf")
        self.assertIsNone(normalized[0]["page"])


if __name__ == "__main__":
    unittest.main()
