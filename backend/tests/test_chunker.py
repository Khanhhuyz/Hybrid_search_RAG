"""
Unit tests for TextChunker service using standard unittest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from app.services.chunker import TextChunker


class TestTextChunker(unittest.TestCase):

    def test_chunker_basic(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a simple text document that needs to be chunked properly."
        chunks = chunker.chunk_document(text, document_id="doc1", document_filename="test.txt")

        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0]["document_id"], "doc1")
        self.assertEqual(chunks[0]["document_filename"], "test.txt")
        self.assertIn("content", chunks[0])

    def test_chunker_section_splitting(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        text = """# Introduction
This is the introduction section of the document.

## Architecture
This section describes the architecture details of the system.
"""
        chunks = chunker.chunk_document(text, document_id="doc2", document_filename="arch.md")

        self.assertGreaterEqual(len(chunks), 2)
        sections = [c["section"] for c in chunks]
        self.assertTrue("Introduction" in sections or "Architecture" in sections)

    def test_chunker_estimate_tokens(self):
        chunker = TextChunker()
        tokens = chunker._estimate_tokens("Hello World!")
        self.assertGreaterEqual(tokens, 1)


if __name__ == "__main__":
    unittest.main()

