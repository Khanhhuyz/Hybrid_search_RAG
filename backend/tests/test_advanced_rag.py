import unittest

from app.services.chunker import TextChunker
from app.services.grounding import GroundednessVerifier
from app.services.hybrid_retriever import BM25Retriever, reciprocal_rank_fusion


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


if __name__ == "__main__":
    unittest.main()
