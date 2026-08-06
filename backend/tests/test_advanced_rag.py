import unittest
import json
import tempfile
from pathlib import Path

from app.services.chunker import TextChunker
from app.services.grounding import GroundednessVerifier, ConfidenceCalibrator
from app.services.answer_processor import AnswerProcessor
from app.services.context_builder import ContextBuilder
from app.services.hybrid_retriever import BM25Retriever, reciprocal_rank_fusion
from app.services.rag_pipeline import RAGPipeline


class TestAdvancedRetrieval(unittest.TestCase):
    def test_bm25_handles_unicode_and_exact_identifiers(self):
        corpus = [
            {"id": "a", "content": "Hợp đồng FSOFT-2026 có hiệu lực tháng tám."},
            {"id": "b", "content": "Thông tin chung về doanh nghiệp."},
        ]
        result = BM25Retriever().search("FSOFT-2026", corpus, 2)
        self.assertEqual(result[0]["id"], "a")

    def test_rrf_fuses_independent_lists(self):
        fused = reciprocal_rank_fusion({
            "dense": [{"id": "a"}, {"id": "b"}],
            "sparse": [{"id": "b"}, {"id": "c"}],
        })
        self.assertEqual(fused[0]["id"], "b")
        self.assertEqual(set(fused[0]["retrieval_sources"]), {"dense", "sparse"})

    def test_page_aware_parent_child_chunks(self):
        chunks = TextChunker(chunk_size=80, chunk_overlap=0).chunk_pages(
            [{"page_number": 7, "text": "# Chính sách\n" + "Nội dung dài. " * 30, "source": "native"}],
            "doc", "policy.pdf",
        )
        self.assertTrue(chunks)
        self.assertTrue(all(item["page_number"] == 7 for item in chunks))
        self.assertTrue(all(item["parent_id"] and item["parent_content"] for item in chunks))

    def test_toc_is_structural_and_not_normal_text(self):
        chunks = TextChunker(chunk_size=200, chunk_overlap=0).chunk_pages(
            [{
                "page_number": 2,
                "text": "Table of Contents\nChapter 1 Introduction 1\nChapter 2 Retrieval 20\n"
                        "Chapter 3 Graphs 40\nChapter 4 Evaluation 60\n"
                        "Chapter 5 Deployment 80\nChapter 6 Appendix 100",
                "source": "native",
            }],
            "doc", "book.pdf",
        )
        self.assertTrue(chunks)
        self.assertTrue(all(item["chunk_type"] == "toc" for item in chunks))
        self.assertTrue(all(item["metadata"]["is_toc"] for item in chunks))

    def test_heading_hierarchy_survives_page_boundaries(self):
        chunks = TextChunker(chunk_size=500, chunk_overlap=0).chunk_pages(
            [
                {"page_number": 1, "text": "# Platform\n## Retrieval\nDense retrieval details.", "source": "native"},
                {"page_number": 2, "text": "### Reranking\nCross encoder details.", "source": "native"},
            ],
            "doc", "architecture.md",
        )
        reranking = next(item for item in chunks if item["section"] == "Reranking")
        self.assertEqual(reranking["metadata"]["heading_path"], ["Platform", "Retrieval", "Reranking"])
        self.assertEqual(reranking["metadata"]["chapter"], "Platform")

    def test_pdf_style_uppercase_heading_is_structural(self):
        chunks = TextChunker(chunk_size=500, chunk_overlap=0).chunk_pages(
            [{"page_number": 1, "text": "RETRIEVAL ARCHITECTURE\nDense retrieval details.", "source": "native"}],
            "doc", "architecture.pdf",
        )
        self.assertEqual(chunks[0]["metadata"]["heading_path"], ["RETRIEVAL ARCHITECTURE"])
        self.assertEqual(chunks[0]["metadata"]["document_filename"], "architecture.pdf")

    def test_global_diversity_covers_documents_and_sections(self):
        ranked = [
            {"id": "a1", "document_id": "a", "section": "One"},
            {"id": "a2", "document_id": "a", "section": "One"},
            {"id": "a3", "document_id": "a", "section": "Two"},
            {"id": "b1", "document_id": "b", "section": "Intro"},
        ]
        result = RAGPipeline._diversify_global_results(ranked, 3)
        self.assertEqual({item["document_id"] for item in result}, {"a", "b"})
        self.assertIn("a3", {item["id"] for item in result})

    def test_context_reports_only_exact_spans_visible_to_the_model(self):
        builder = ContextBuilder(max_context_tokens=80)
        result = builder.build(
            semantic_results=[
                {"id": "a", "content": "A" * 1000, "document_filename": "a.pdf"},
                {"id": "b", "content": "B" * 1000, "document_filename": "b.pdf"},
            ],
            graph_context=[],
        )
        self.assertEqual(len(result["semantic_sources"]), 1)
        visible = result["semantic_sources"][0]["content"]
        self.assertTrue(visible)
        self.assertLess(len(visible), 1000)
        self.assertIn(visible, result["semantic_context"])


class TestGrounding(unittest.TestCase):
    def test_claim_requires_matching_citation_evidence(self):
        result = GroundednessVerifier().verify(
            "Alice quản lý Project Alpha [S1].",
            [{"support_text": "Alice quản lý Project Alpha."}],
        )
        self.assertEqual(result["groundedness_score"], 1.0)

    def test_uncited_claim_is_not_grounded(self):
        result = GroundednessVerifier().verify("Alice quản lý Project Alpha.", [])
        self.assertEqual(result["groundedness_score"], 0.0)

    def test_numeric_contradiction_is_rejected(self):
        result = GroundednessVerifier().verify(
            "The model achieved 99% accuracy [S1].",
            [{"support_text": "The model achieved 50% accuracy."}],
        )
        self.assertFalse(result["claims"][0]["supported"])
        self.assertEqual(result["claims"][0]["reason"], "numeric_mismatch")

    def test_relation_reversal_is_rejected(self):
        result = GroundednessVerifier().verify(
            "Alice manages Bob [S1].",
            [{"support_text": "Bob manages Alice."}],
        )
        self.assertFalse(result["claims"][0]["supported"])

    def test_negation_contradiction_is_rejected(self):
        result = GroundednessVerifier().verify(
            "Paris is not in France [S1].",
            [{"support_text": "Paris is in France."}],
        )
        self.assertFalse(result["claims"][0]["supported"])
        self.assertEqual(result["claims"][0]["reason"], "negation_mismatch")

    def test_one_supporting_citation_is_not_poisoned_by_an_unrelated_one(self):
        result = GroundednessVerifier().verify(
            "The system achieved 99% accuracy [S1][S2].",
            [
                {"support_text": "The system achieved 99% accuracy."},
                {"support_text": "A separate baseline did not exceed 40%."},
            ],
        )
        self.assertEqual(result["groundedness_score"], 1.0)
        self.assertTrue(result["claims"][0]["supported"])

    def test_unsupported_answer_is_replaced_before_delivery(self):
        processed = AnswerProcessor().process(
            "The model achieved 99% accuracy [S1].",
            [{"support_text": "The model achieved 50% accuracy."}],
            1, 0, "semantic", question="What accuracy?", evidence_score=0.9,
        )
        self.assertIn("could not find enough evidence", processed["answer"].lower())
        self.assertEqual(processed["confidence_score"], 0.0)

    def test_calibrator_loads_fitted_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"coefficients": [2.0, 3.0], "intercept": -2.0}), encoding="utf-8")
            calibrator = ConfidenceCalibrator(path)
            probability, calibrated = calibrator.predict(1.0, 1.0)
            self.assertTrue(calibrated)
            self.assertGreater(probability, 0.9)


if __name__ == "__main__":
    unittest.main()
