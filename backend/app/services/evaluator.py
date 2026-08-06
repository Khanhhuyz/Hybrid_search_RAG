"""Deterministic retrieval and answer metrics for offline GraphRAG evaluation."""
import re
import unicodedata
import math
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
        dcg = sum((1.0 if item in relevant else 0.0) / math.log2(rank + 1) for rank, item in enumerate(retrieved, 1))
        ideal_hits = min(len(relevant), len(retrieved))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "hit_rate": float(bool(hits)),
            "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
            "ndcg": round(dcg / idcg, 4) if idcg else 0.0,
        }

    @staticmethod
    def set_metrics(predicted: List[str], expected: List[str], prefix: str) -> Dict[str, float]:
        predicted_set, expected_set = set(predicted), set(expected)
        hits = len(predicted_set & expected_set)
        precision = hits / len(predicted_set) if predicted_set else (1.0 if not expected_set else 0.0)
        recall = hits / len(expected_set) if expected_set else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {f"{prefix}_precision": round(precision, 4), f"{prefix}_recall": round(recall, 4), f"{prefix}_f1": round(f1, 4)}

    @staticmethod
    def confidence_metrics(cases: List[Dict], bins: int = 10) -> Dict[str, float]:
        labelled = [case for case in cases if "confidence_score" in case and "is_correct" in case]
        if not labelled:
            return {}
        probabilities = [max(0.0, min(1.0, float(case["confidence_score"]))) for case in labelled]
        labels = [float(bool(case["is_correct"])) for case in labelled]
        brier = mean((probability - label) ** 2 for probability, label in zip(probabilities, labels))
        ece = 0.0
        for index in range(bins):
            low, high = index / bins, (index + 1) / bins
            members = [
                i for i, value in enumerate(probabilities)
                if low <= value < high or (index == bins - 1 and value == 1.0)
            ]
            if members:
                accuracy = mean(labels[i] for i in members)
                confidence = mean(probabilities[i] for i in members)
                ece += len(members) / len(labels) * abs(accuracy - confidence)
        return {
            "confidence_brier": round(brier, 4),
            "confidence_ece": round(ece, 4),
            "confidence_cases": len(labelled),
        }

    @classmethod
    def evaluate_case(cls, case: Dict) -> Dict:
        metrics = cls.retrieval_metrics(case["retrieved_chunk_ids"], case["relevant_chunk_ids"])
        metrics["answer_token_f1"] = round(
            cls.answer_token_f1(case.get("answer", ""), case.get("reference_answer", "")),
            4,
        )
        citation_ids = [str(item.get("source_id", item.get("chunk_id", ""))) for item in case.get("citations", [])]
        metrics.update(cls.set_metrics(citation_ids, case.get("relevant_source_ids", []), "citation"))
        metrics["no_answer_accuracy"] = float(bool(case.get("predicted_no_answer")) == bool(case.get("expected_no_answer")))
        metrics.update(cls.set_metrics(case.get("extracted_entities", []), case.get("expected_entities", []), "entity"))
        metrics.update(cls.set_metrics(case.get("extracted_relations", []), case.get("expected_relations", []), "relation"))
        return {"question": case["question"], "metrics": metrics}

    @classmethod
    def evaluate_batch(cls, cases: Iterable[Dict]) -> Dict:
        case_list = list(cases)
        results = [cls.evaluate_case(case) for case in case_list]
        metric_names = tuple(results[0]["metrics"].keys()) if results else (
            "precision", "recall", "f1", "hit_rate", "reciprocal_rank", "ndcg", "answer_token_f1",
            "citation_precision", "citation_recall", "citation_f1", "no_answer_accuracy",
            "entity_precision", "entity_recall", "entity_f1", "relation_precision", "relation_recall", "relation_f1",
        )
        aggregate = {
            name: round(mean(result["metrics"][name] for result in results), 4)
            for name in metric_names
        } if results else {name: 0.0 for name in metric_names}
        aggregate.update(cls.confidence_metrics(case_list))
        return {"total_cases": len(results), "aggregate": aggregate, "cases": results}
