import json
import tempfile
import unittest
from pathlib import Path

from src.citation import validate_citations
from src.feedback import FeedbackStore
from src.memory import MemoryStore
from src.models import DocumentChunk, RetrievalResult


class CitationValidatorTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            RetrievalResult(
                chunk=DocumentChunk(
                    chunk_id="chunk-1",
                    source="paper.pdf",
                    page=5,
                    text="氧化还原条件和沉积速率影响多金属结核形成。",
                ),
                score=0.8,
            )
        ]

    def test_valid_citation_and_source_page_pass(self):
        answer = "氧化还原条件会影响多金属结核形成。[1]\n\n来源：\n[1] paper.pdf 第 5 页"
        validation = validate_citations(answer, self.results)
        self.assertTrue(validation.passed)
        self.assertEqual(validation.valid_citations, [1])

    def test_fabricated_page_is_rejected(self):
        answer = "氧化还原条件会影响多金属结核形成。[1]\n\n来源：\n[1] paper.pdf 第 99 页"
        validation = validate_citations(answer, self.results)
        self.assertFalse(validation.passed)
        self.assertTrue(any("页码" in issue for issue in validation.invalid_citations))


class LayeredMemoryTests(unittest.TestCase):
    def test_compaction_and_long_term_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "state.db")
            store.append_turn("s1", "第一个问题", "第一个回答", True, 2, 500, 1000)
            store.append_turn("s1", "第二个问题", "第二个回答", True, 2, 500, 1000)
            store.add_long_term("用户要求所有回答使用中文", scope="kb1")

            context = store.get_context("s1", "回答语言要求", 2, 4, scope="kb1")
            self.assertEqual(len(context.history), 2)
            self.assertIn("第一个问题", context.summary)
            self.assertIn("所有回答使用中文", context.long_term[0])

    def test_unvalidated_turn_does_not_enter_agent_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "state.db")
            store.append_turn("s1", "错误问题", "未经校验的回答", False, 12, 6000, 4000)
            snapshot = store.snapshot("s1")
            self.assertEqual(snapshot["short_term"], [])
            self.assertEqual(snapshot["summary"], "")


class FeedbackLoopTests(unittest.TestCase):
    def test_negative_feedback_creates_review_item_and_eval_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.db"
            bad_cases = Path(tmp) / "bad_cases.jsonl"
            store = FeedbackStore(db_path, bad_cases)
            item = store.submit(
                trace_id="trace-1",
                session_id="session-1",
                rating=-1,
                question="控制因素有哪些？",
                answer="原回答",
                issue_type="事实错误",
                correction="正确答案",
                evidence=[{"source": "paper.pdf", "page": 5}],
            )
            self.assertEqual(item["status"], "pending")
            case = json.loads(bad_cases.read_text(encoding="utf-8").strip())
            self.assertEqual(case["question"], "控制因素有哪些？")
            self.assertEqual(case["expected_answer"], "正确答案")
            self.assertEqual(case["expected_sources"][0]["page"], 5)

            reviewed = store.update_status(item["feedback_id"], "resolved")
            self.assertEqual(reviewed["status"], "resolved")


if __name__ == "__main__":
    unittest.main()
