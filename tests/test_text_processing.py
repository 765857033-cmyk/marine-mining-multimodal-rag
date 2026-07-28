import unittest

from src.text_processing import clean_pdf_pages, pages_to_chunks


class TextProcessingTests(unittest.TestCase):
    def test_clean_pdf_pages_removes_repeated_footer(self):
        pages = [
            (1, "Paper title\nUseful content about polymetallic nodules.\nJournal footer 2024\n1"),
            (2, "Paper title\nMore useful content about cobalt crust.\nJournal footer 2024\n2"),
            (3, "Paper title\nHydrothermal sulfide content.\nJournal footer 2024\n3"),
        ]
        cleaned = clean_pdf_pages(pages)
        joined = "\n".join(text for _, text in cleaned)
        self.assertNotIn("Journal footer", joined)
        self.assertIn("Useful content", joined)

    def test_pages_to_chunks_keeps_sources(self):
        chunks = pages_to_chunks(
            [(1, "多金属结核成矿环境包括氧化还原条件、沉积速率和金属来源。" * 20)],
            source="demo.pdf",
            chunk_size=300,
            overlap=40,
        )
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source, "demo.pdf")


if __name__ == "__main__":
    unittest.main()
