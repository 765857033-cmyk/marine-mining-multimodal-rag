import os
import unittest
from unittest.mock import patch

from src.config import AppConfig
from src.llm import resolve_llm_endpoint


class LlmApiBackendTests(unittest.TestCase):
    def test_resolves_oceangpt_api_backend(self):
        with patch.dict(os.environ, {"OCEANGPT_API_KEY": "test-key"}, clear=False):
            endpoint = resolve_llm_endpoint(
                AppConfig(
                    llm_backend="ocean-gpt-api",
                    oceangpt_base_url="https://example.com/v1",
                    oceangpt_model="OceanGPT-test",
                )
            )
        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint.backend, "ocean-gpt-api")
        self.assertEqual(endpoint.model, "OceanGPT-test")
        self.assertEqual(endpoint.base_url, "https://example.com/v1")

    def test_none_backend_disables_remote_llm(self):
        endpoint = resolve_llm_endpoint(AppConfig(llm_backend="none"))
        self.assertIsNone(endpoint)


if __name__ == "__main__":
    unittest.main()
