"""Unit tests for deterministic offline evaluation metrics."""
import unittest

from app.services.evaluator import Evaluator


class TestEvaluator(unittest.TestCase):
    def test_retrieval_metrics(self):
        metrics = Evaluator.retrieval_metrics(["c1", "c2", "c3"], ["c2", "c4"])
        self.assertAlmostEqual(metrics["precision"], 1 / 3, places=4)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["hit_rate"], 1.0)

    def test_answer_token_f1_is_unicode_aware(self):
        score = Evaluator.answer_token_f1(
            "GraphRAG kết hợp đồ thị tri thức.",
            "Đồ thị tri thức được GraphRAG kết hợp.",
        )
        self.assertGreater(score, 0.8)

    def test_batch_aggregate(self):
        report = Evaluator.evaluate_batch([{
            "question": "q",
            "retrieved_chunk_ids": ["c1"],
            "relevant_chunk_ids": ["c1"],
            "answer": "same answer",
            "reference_answer": "same answer",
        }])
        self.assertEqual(report["total_cases"], 1)
        self.assertEqual(report["aggregate"]["f1"], 1.0)
        self.assertEqual(report["aggregate"]["answer_token_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
