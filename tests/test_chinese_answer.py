import unittest

from src.llm import AnswerGenerator
from src.config import AppConfig
from src.models import DocumentChunk, RetrievalResult


class ChineseAnswerTests(unittest.TestCase):
    def test_fallback_answer_summarizes_english_evidence_in_chinese(self):
        result = RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="en-1",
                source="english.pdf",
                page=2,
                text=(
                    "Polymetallic nodules are controlled by redox condition, "
                    "sedimentation rate, and metal source."
                ),
            ),
            score=0.8,
            rerank_score=0.8,
        )
        answer = AnswerGenerator(AppConfig(llm_backend="none")).generate("多金属结核受哪些因素控制？", [result])
        self.assertIn("英文文献证据表明", answer)
        self.assertIn("多金属结核", answer)
        self.assertIn("氧化还原条件", answer)
        self.assertNotIn("Polymetallic nodules are controlled", answer)


if __name__ == "__main__":
    unittest.main()
