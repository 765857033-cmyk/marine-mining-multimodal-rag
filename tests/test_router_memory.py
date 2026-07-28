import unittest

from src.agent import enrich_followup_question, normalize_history
from src.llm import format_history, parse_router_json
from src.models import ConversationTurn


class RouterMemoryTests(unittest.TestCase):
    def test_parse_router_json_from_fenced_output(self):
        data = parse_router_json(
            '```json\n{"route":"rag","need_retrieval":true,'
            '"rewritten_query":"多金属结核 控制因素","reason":"需要文献证据","confidence":0.91}\n```'
        )
        self.assertEqual(data["route"], "rag")
        self.assertTrue(data["need_retrieval"])
        self.assertEqual(data["rewritten_query"], "多金属结核 控制因素")
        self.assertAlmostEqual(data["confidence"], 0.91)

    def test_format_history_uses_recent_turns(self):
        history = [
            ConversationTurn("user", "第一个问题"),
            ConversationTurn("assistant", "第一个回答"),
            ConversationTurn("user", "第二个问题"),
        ]
        text = format_history(history, max_turns=1)
        self.assertNotIn("第一个问题", text)
        self.assertIn("第一个回答", text)
        self.assertIn("第二个问题", text)

    def test_followup_question_is_enriched_with_memory(self):
        history = normalize_history(
            [
                {"role": "user", "content": "多金属结核成矿受哪些因素控制？"},
                {"role": "assistant", "content": "主要与氧化还原条件和沉积速率有关。"},
            ],
            max_turns=3,
        )
        enriched = enrich_followup_question("继续解释这些因素", history)
        self.assertIn("继续解释这些因素", enriched)
        self.assertIn("多金属结核成矿", enriched)


if __name__ == "__main__":
    unittest.main()
