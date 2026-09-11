import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig
from src.multi_agents import MultiAgentRagSystem
from src.vector_store import EmbeddingProvider, VectorIndex


class MultiAgentTests(unittest.TestCase):
    def test_multi_agent_system_excludes_incremental_update_agent(self):
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
            index = VectorIndex(EmbeddingProvider("hashing"), backend="chroma")
            system = MultiAgentRagSystem(config, index)

        self.assertEqual(
            system.agent_names,
            ["DocParserAgent", "KnowledgeExtractAgent", "QAAgent"],
        )
        self.assertNotIn("KnowledgeUpdateAgent", system.agent_names)


if __name__ == "__main__":
    unittest.main()
