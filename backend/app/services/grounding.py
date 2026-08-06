"""Claim-level citation entailment checks and learned confidence calibration."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings
from app.services.hybrid_retriever import tokenize


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "from", "with", "by", "and", "or",
    "this", "that", "these", "those", "it", "its", "as", "has", "have", "had",
    "là", "của", "và", "hoặc", "trong", "trên", "tại", "cho", "từ", "với",
    "được", "có", "này", "đó", "những", "các", "một",
}
NEGATIONS = {
    "not", "no", "never", "neither", "without", "cannot", "can't", "isn't",
    "aren't", "wasn't", "weren't", "không", "chưa", "chẳng", "chả",
}
NO_ANSWER_PATTERNS = (
    "could not find enough evidence", "cannot find enough evidence",
    "context does not contain", "not enough information", "insufficient evidence",
    "chưa tìm thấy đủ bằng chứng", "không có đủ thông tin", "không đủ bằng chứng",
    "không có thông tin trong", "không thể xác định",
)


def _term(value: str) -> str:
    """Small deterministic normalizer for common English inflections."""
    value = value.casefold().strip("._-")
    if len(value) > 5 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("es"):
        return value[:-2]
    if len(value) > 3 and value.endswith("s"):
        return value[:-1]
    return value


class ConfidenceCalibrator:
    """Load a Platt-style logistic model fitted on reviewed answer outcomes."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.CALIBRATION_FILE)
        self.coefficients: List[float] | None = None
        self.intercept = 0.0
        self._load()

    def _load(self):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            coefficients = [float(value) for value in payload["coefficients"]]
            if len(coefficients) != 2:
                raise ValueError("expected two calibration coefficients")
            self.coefficients = coefficients
            self.intercept = float(payload["intercept"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.coefficients = None

    @property
    def is_calibrated(self) -> bool:
        return self.coefficients is not None

    def predict(self, evidence_score: float, groundedness: float) -> Tuple[float, bool]:
        evidence = max(0.0, min(1.0, float(evidence_score)))
        grounded = max(0.0, min(1.0, float(groundedness)))
        if self.coefficients:
            logit = self.intercept + self.coefficients[0] * evidence + self.coefficients[1] * grounded
            return 1 / (1 + math.exp(-max(-30.0, min(30.0, logit)))), True
        # Without reviewed labels, never emit a deceptively high probability.
        # The fitting script replaces this safe fallback with a learned model.
        raw = 0.45 * evidence + 0.55 * grounded
        return min(raw, settings.NO_ANSWER_CONFIDENCE_THRESHOLD - 0.01), False


class GroundednessVerifier:
    def __init__(self):
        self.calibrator = ConfidenceCalibrator()

    @staticmethod
    def is_no_answer(answer: str) -> bool:
        lowered = (answer or "").casefold()
        return any(pattern in lowered for pattern in NO_ANSWER_PATTERNS)

    def split_claims(self, answer: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", answer or "")
        return [part.strip() for part in parts if len(tokenize(part)) >= 3]

    @staticmethod
    def _numbers(text: str) -> set[str]:
        return set(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?(?!\w)", text or ""))

    @staticmethod
    def _has_negation(text: str) -> bool:
        return any(token.casefold() in NEGATIONS for token in tokenize(text))

    @classmethod
    def _support_detail(cls, claim: str, evidence: str) -> Tuple[float, str | None]:
        clean_claim = re.sub(r"\[[SC]\d+\]", "", claim)
        claim_numbers = cls._numbers(clean_claim)
        if claim_numbers and not claim_numbers.issubset(cls._numbers(evidence)):
            return 0.0, "numeric_mismatch"
        if cls._has_negation(clean_claim) != cls._has_negation(evidence):
            return 0.0, "negation_mismatch"

        claim_terms = [
            _term(token) for token in tokenize(clean_claim)
            if len(token) > 1 and token.casefold() not in STOPWORDS and token.casefold() not in NEGATIONS
        ]
        evidence_sequence = [
            _term(token) for token in tokenize(evidence)
            if len(token) > 1 and token.casefold() not in STOPWORDS and token.casefold() not in NEGATIONS
        ]
        evidence_terms = set(evidence_sequence)
        if not claim_terms:
            return 0.0, "empty_claim"
        lexical_support = sum(term in evidence_terms for term in claim_terms) / len(claim_terms)
        # Token bags accept relation reversal ("Alice manages Bob" vs "Bob
        # manages Alice"). Require the supported terms to retain their order.
        previous = [0] * (len(evidence_sequence) + 1)
        for claim_term in claim_terms:
            current = [0]
            for index, evidence_term in enumerate(evidence_sequence, 1):
                current.append(
                    previous[index - 1] + 1
                    if claim_term == evidence_term
                    else max(previous[index], current[-1])
                )
            previous = current
        ordered_support = previous[-1] / len(claim_terms)
        support = min(lexical_support, ordered_support)
        reason = None if support >= settings.GROUNDEDNESS_THRESHOLD else "insufficient_entailment"
        return support, reason

    @classmethod
    def _support_score(cls, claim: str, evidence: str) -> float:
        return cls._support_detail(claim, evidence)[0]

    def verify(self, answer: str, citations: List[Dict]) -> Dict:
        by_id = {f"S{index + 1}": citation for index, citation in enumerate(citations)}
        claims, details = self.split_claims(answer), []
        for claim in claims:
            refs = re.findall(r"\[(S\d+)\]", claim)
            valid_refs = list(dict.fromkeys(ref for ref in refs if ref in by_id))
            invalid_refs = [ref for ref in refs if ref not in by_id]
            candidates = [
                self._support_detail(
                    claim,
                    by_id[ref].get("support_text") or by_id[ref].get("excerpt", ""),
                )
                for ref in valid_refs
            ]
            if not candidates:
                support, reason = 0.0, "missing_citation"
            else:
                # One genuinely supporting source is sufficient; unrelated text in
                # another citation must not create a spurious polarity mismatch.
                supported_candidates = [item for item in candidates if item[1] is None]
                support, reason = (
                    max(supported_candidates, key=lambda item: item[0])
                    if supported_candidates
                    else max(candidates, key=lambda item: item[0])
                )
            if invalid_refs:
                reason = "invalid_citation"
            details.append({
                "claim": claim,
                "citations": valid_refs,
                "invalid_citations": invalid_refs,
                "support_score": round(support, 3),
                "reason": reason,
                "supported": bool(valid_refs and not invalid_refs and reason is None),
            })
        supported = sum(item["supported"] for item in details)
        coverage = supported / len(details) if details else 0.0
        return {"groundedness_score": round(coverage, 3), "claims": details}

    def confidence(self, evidence_score: float, groundedness: float) -> Tuple[float, bool]:
        return self.calibrator.predict(evidence_score, groundedness)

    def calibrated_confidence(self, evidence_score: float, groundedness: float) -> float:
        return self.confidence(evidence_score, groundedness)[0]
