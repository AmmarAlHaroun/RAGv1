import unittest

from rag.chunker import chunk_text_with_metadata


class ChunkerTests(unittest.TestCase):
    def test_plain_pdf_headings_get_section_titles(self):
        text = "Document Title\nSection One\nThis is body text with enough words to identify it as body content."
        chunks = chunk_text_with_metadata(text, chunk_size=100, overlap=0, plain_pdf=True)
        self.assertEqual(chunks[0]["section_title"], "Document Title")
        self.assertEqual(chunks[1]["section_title"], "Section One")

    def test_txt_behavior_does_not_enable_plain_pdf_heading_detection(self):
        text = "Document Title\nSection One\nThis is body text with enough words to identify it as body content."
        chunks = chunk_text_with_metadata(text, chunk_size=100, overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section_title"], "")


if __name__ == "__main__":
    unittest.main()
