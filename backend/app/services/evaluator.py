"""Deterministic retrieval and answer metrics for offline GraphRAG evaluation."""
import re
import unicodedata
from statistics import mean
from typing import Dict, Iterable, List


class Evaluator:
    """Evaluate labelled examples without requiring an LLM or external service."""

    @staticmethod
    def _tokens(text: str) -> List[str]:
        normalized = unicodedata.normalize("NFKC", text or "").casefold()
        return re.findall(r"\w+", normalized, flags=re.UNICODE)

    @classmethod
    def answer_token_f1(cls, prediction: str, reference: str) -> float:
        predicted = cls._tokens(prediction)
        expected = cls._tokens(reference)
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0

        predicted_counts = {token: predicted.count(token) for token in set(predicted)}
        expected_counts = {token: expected.count(token) for token in set(expected)}
        overlap = sum(
            min(count, expected_counts.get(token, 0))
            for token, count in predicted_counts.items()
        )
        if overlap == 0:
            return 0.0
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def retrieval_metrics(retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
        retrieved = list(dict.fromkeys(retrieved_ids))
        relevant = set(relevant_ids)
        hits = [item for item in retrieved if item in relevant]
        precision = len(hits) / len(retrieved) if retrieved else 0.0
        recall = len(hits) / len(relevant) if relevant else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        first_rank = next((i for i, item in enumerate(retrieved, 1) if item in relevant), None)
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hit_rate": float(bool(hits)),
            "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
        }

    @classmethod
    def evaluate_case(cls, case: Dict) -> Dict:
        metrics = cls.retrieval_metrics(case["retrieved_chunk_ids"], case["relevant_chunk_ids"])
        metrics["answer_token_f1"] = round(
            cls.answer_token_f1(case.get("answer", ""), case.get("reference_answer", "")),
            4,
        )
        return {"question": case["question"], "metrics": metrics}

    @classmethod
    def evaluate_batch(cls, cases: Iterable[Dict]) -> Dict:
        results = [cls.evaluate_case(case) for case in cases]
        metric_names = (
            "precision", "recall", "f1", "hit_rate", "reciprocal_rank", "answer_token_f1"
        )
        aggregate = {
            name: round(mean(result["metrics"][name] for result in results), 4)
            for name in metric_names
        } if results else {name: 0.0 for name in metric_names}
        return {"total_cases": len(results), "aggregate": aggregate, "cases": results}
