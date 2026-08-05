"""Claim-level citation validation and conservative confidence calibration."""
from __future__ import annotations

import math
import re
from typing import Dict, List

from app.config import settings
from app.services.hybrid_retriever import tokenize


class GroundednessVerifier:
    def split_claims(self, answer: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+|\n+", answer or "")
        return [part.strip() for part in parts if len(tokenize(part)) >= 3]

    @staticmethod
    def _support_score(claim: str, evidence: str) -> float:
        claim_terms = [term for term in tokenize(re.sub(r"\[[SC]\d+\]", "", claim)) if len(term) > 1]
        evidence_terms = set(tokenize(evidence))
        if not claim_terms:
            return 1.0
        return sum(term in evidence_terms for term in claim_terms) / len(claim_terms)

    def verify(self, answer: str, citations: List[Dict]) -> Dict:
        by_id = {f"S{index + 1}": citation for index, citation in enumerate(citations)}
        claims, details = self.split_claims(answer), []
        for claim in claims:
            refs = re.findall(r"\[(S\d+)\]", claim)
            valid_refs = [ref for ref in refs if ref in by_id]
            evidence = " ".join(
                by_id[ref].get("support_text") or by_id[ref].get("excerpt", "")
                for ref in valid_refs
            )
            support = self._support_score(claim, evidence) if valid_refs else 0.0
            details.append({
                "claim": claim,
                "citations": valid_refs,
                "support_score": round(support, 3),
                "supported": bool(valid_refs and support >= settings.GROUNDEDNESS_THRESHOLD),
            })
        supported = sum(item["supported"] for item in details)
        coverage = supported / len(details) if details else 0.0
        return {"groundedness_score": round(coverage, 3), "claims": details}

    @staticmethod
    def calibrated_confidence(evidence_score: float, groundedness: float) -> float:
        combined = 0.45 * max(0.0, min(1.0, evidence_score)) + 0.55 * groundedness
        return 1 / (1 + math.exp(-settings.CALIBRATION_SLOPE * (combined - settings.CALIBRATION_MIDPOINT)))
