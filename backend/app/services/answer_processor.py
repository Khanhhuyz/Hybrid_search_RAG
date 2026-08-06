"""
Answer Processor
Post-processing pipeline for LLM-generated answers:
- Citation validation
- Confidence scoring
- Language consistency check
"""
import re
import logging
from typing import Dict, List, Optional
from app.config import settings
from app.services.grounding import GroundednessVerifier

logger = logging.getLogger(__name__)


class AnswerProcessor:
    """Post-process LLM answers for quality assurance."""

    def __init__(self):
        self.grounding = GroundednessVerifier()

    def process(
        self,
        answer: str,
        citations: List[Dict],
        semantic_chunks_used: int,
        graph_nodes_used: int,
        retrieval_mode: str,
        question: str = "",
        evidence_score: float = 0.0,
    ) -> Dict:
        """
        Full post-processing pipeline.

        Returns:
            {
                "answer": str (cleaned),
                "confidence_score": float,
                "citation_valid": bool,
                "warnings": List[str],
            }
        """
        warnings = []
        cleaned_answer = self._clean_answer(answer)

        # Explicit no-answer responses are safe and must not receive confidence
        # merely because retrieval returned several unrelated chunks.
        if self.grounding.is_no_answer(cleaned_answer):
            return {
                "answer": self._no_answer(question),
                "confidence_score": 0.0,
                "confidence_calibrated": self.grounding.calibrator.is_calibrated,
                "citation_valid": True,
                "warnings": ["The answer abstained because evidence was insufficient"],
                "groundedness_score": 0.0,
                "claim_support": [],
            }

        verification = self.grounding.verify(cleaned_answer, citations)
        citation_valid, citation_warnings = self._validate_citations(
            cleaned_answer, citations, verification["claims"]
        )
        warnings.extend(citation_warnings)
        unsupported = [item for item in verification["claims"] if not item["supported"]]
        if unsupported:
            warnings.append(f"{len(unsupported)} claim(s) lack sufficient cited evidence")
        confidence, confidence_calibrated = self.grounding.confidence(
            evidence_score=evidence_score,
            groundedness=verification["groundedness_score"],
        )
        if not confidence_calibrated:
            warnings.append("Confidence is using the conservative fallback; fit a calibration model")

        insufficient = (
            evidence_score < settings.RETRIEVAL_MIN_EVIDENCE_SCORE
            or not verification["claims"]
            or bool(unsupported)
            or not citation_valid
        )
        if evidence_score < settings.RETRIEVAL_MIN_EVIDENCE_SCORE:
            warnings.append("Retrieved evidence is insufficient for a reliable answer")
        if settings.GROUNDING_ENFORCED and insufficient:
            cleaned_answer = self._no_answer(question)
            confidence = 0.0

        return {
            "answer": cleaned_answer,
            "confidence_score": round(confidence, 3),
            "confidence_calibrated": confidence_calibrated,
            "citation_valid": citation_valid,
            "warnings": warnings,
            "groundedness_score": verification["groundedness_score"],
            "claim_support": verification["claims"],
        }

    def _validate_citations(
        self, answer: str, citations: List[Dict], claims: Optional[List[Dict]] = None
    ) -> tuple:
        """Check that all citation references in the answer exist in the citation list."""
        warnings = []

        # Find all citation references like [S1], [S2], [C1] in the answer
        referenced = set(re.findall(r"\[([SC]\d+)\]", answer))
        available_source_ids = {f"S{i+1}" for i in range(len(citations))}

        # Check for non-existent citations
        invalid = referenced - available_source_ids
        if invalid:
            warnings.append(
                f"Answer references non-existent sources: {', '.join(sorted(invalid))}"
            )

        claims = claims or []
        missing = [item for item in claims if not item.get("citations")]
        if missing:
            warnings.append(f"{len(missing)} factual claim(s) have no citation")
        is_valid = len(invalid) == 0 and not missing and bool(claims)
        return is_valid, warnings

    @staticmethod
    def _no_answer(question: str) -> str:
        vietnamese = any(
            char in (question or "").casefold()
            for char in "ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ"
        )
        return (
            "Tôi chưa tìm thấy đủ bằng chứng trong tài liệu để trả lời câu hỏi này."
            if vietnamese else
            "I could not find enough evidence in the documents to answer this question."
        )

    def _clean_answer(self, answer: str) -> str:
        """Clean up common LLM output artifacts."""
        # Remove common LLM prefix artifacts
        prefixes_to_remove = [
            "Based on the provided context,",
            "Dựa trên ngữ cảnh được cung cấp,",
            "According to the documents,",
            "Theo tài liệu,",
        ]
        for prefix in prefixes_to_remove:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                # Capitalize first letter
                if answer:
                    answer = answer[0].upper() + answer[1:]

        return answer.strip()
