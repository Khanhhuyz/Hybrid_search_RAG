"""
Unit tests for RRF re-ranking and helper functions in RAGPipeline using unittest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from unittest.mock import MagicMock

try:
    from app.services.rag_pipeline import RAGPipeline
except ModuleNotFoundError:
    sys.modules['qdrant_client'] = MagicMock()
    sys.modules['qdrant_client.http'] = MagicMock()
    sys.modules['qdrant_client.http.models'] = MagicMock()
    from app.services.rag_pipeline import RAGPipeline


class TestRAGPipelineLogic(unittest.TestCase):

    def test_rrf_scoring(self):
        embedder = MagicMock()
        vector_store = MagicMock()
        graph_builder = MagicMock()
        pipeline = RAGPipeline(embedder, vector_store, graph_builder)

        sample_results = [
            {"id": "c1", "score": 0.9, "content": "chunk 1"},
            {"id": "c2", "score": 0.7, "content": "chunk 2"},
            {"id": "c3", "score": 0.95, "content": "chunk 3"},
        ]

        reranked = pipeline._apply_rrf(sample_results, top_k=2)
        self.assertEqual(len(reranked), 2)
        self.assertIn("rrf_score", reranked[0])
        # Check that highest combined rank comes first
        self.assertGreaterEqual(reranked[0]["rrf_score"], reranked[1]["rrf_score"])

    def test_citations_building(self):
        embedder = MagicMock()
        vector_store = MagicMock()
        graph_builder = MagicMock()
        pipeline = RAGPipeline(embedder, vector_store, graph_builder)

        sample_results = [
            {"id": "c1", "document_id": "d1", "document_filename": "f1.pdf", "score": 0.8, "content": "Sample content"},
            {"id": "c1", "document_id": "d1", "document_filename": "f1.pdf", "score": 0.8, "content": "Duplicate chunk"},
        ]

        citations = pipeline._build_citations(sample_results)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["document_id"], "d1")
        self.assertEqual(citations[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()

