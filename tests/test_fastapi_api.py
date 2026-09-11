import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import create_app
from src.config import AppConfig


class FastApiTests(unittest.TestCase):
    def test_health_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                index_dir=Path(tmp) / "index",
                upload_dir=Path(tmp) / "uploads",
                state_db_path=Path(tmp) / "state.db",
                trace_dir=Path(tmp) / "traces",
                feedback_bad_cases_path=Path(tmp) / "bad_cases.jsonl",
                embedding_backend="hashing",
                vector_backend="chroma",
                parser_backend="mineru",
            )
            client = TestClient(create_app(config))

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")

            stats = client.get("/index/stats")
            self.assertEqual(stats.status_code, 200)
            self.assertEqual(stats.json()["parent_chunks"], 0)

    def test_static_react_frontend_is_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                index_dir=Path(tmp) / "index",
                upload_dir=Path(tmp) / "uploads",
                state_db_path=Path(tmp) / "state.db",
                trace_dir=Path(tmp) / "traces",
                feedback_bad_cases_path=Path(tmp) / "bad_cases.jsonl",
                embedding_backend="hashing",
                vector_backend="chroma",
            )
            client = TestClient(create_app(config))
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("React", response.text)
            self.assertIn("海洋矿产 Agentic RAG", response.text)

    def test_chat_requires_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                index_dir=Path(tmp) / "index",
                upload_dir=Path(tmp) / "uploads",
                state_db_path=Path(tmp) / "state.db",
                trace_dir=Path(tmp) / "traces",
                feedback_bad_cases_path=Path(tmp) / "bad_cases.jsonl",
                embedding_backend="hashing",
                vector_backend="chroma",
            )
            client = TestClient(create_app(config))
            response = client.post("/chat", json={"question": ""})
            self.assertEqual(response.status_code, 422)

    def test_memory_feedback_and_trace_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig(
                index_dir=Path(tmp) / "index",
                upload_dir=Path(tmp) / "uploads",
                state_db_path=Path(tmp) / "state.db",
                trace_dir=Path(tmp) / "traces",
                feedback_bad_cases_path=Path(tmp) / "bad_cases.jsonl",
                embedding_backend="hashing",
                vector_backend="chroma",
            )
            client = TestClient(create_app(config))
            chat = client.post(
                "/chat",
                json={"question": "你好", "session_id": "api-session"},
            )
            self.assertEqual(chat.status_code, 200)
            payload = chat.json()
            self.assertTrue(payload["trace_id"])
            self.assertTrue(payload["trace_events"])

            memory = client.get("/memory/api-session")
            self.assertEqual(memory.status_code, 200)
            self.assertEqual(len(memory.json()["short_term"]), 2)

            feedback = client.post(
                "/feedback",
                json={
                    "trace_id": payload["trace_id"],
                    "session_id": "api-session",
                    "rating": -1,
                    "question": "你好",
                    "answer": payload["answer"],
                    "issue_type": "回答不完整",
                },
            )
            self.assertEqual(feedback.status_code, 200)
            self.assertEqual(feedback.json()["status"], "pending")

            trace = client.get(f"/traces/{payload['trace_id']}")
            self.assertEqual(trace.status_code, 200)
            self.assertEqual(trace.json()["trace_id"], payload["trace_id"])


if __name__ == "__main__":
    unittest.main()
