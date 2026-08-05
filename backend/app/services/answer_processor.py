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

        # 1. Citation validation
        citation_valid, citation_warnings = self._validate_citations(answer, citations)
        warnings.extend(citation_warnings)

        # 2. Verify every factual claim against the cited source text, then
        # calibrate confidence from evidence quality instead of answer length.
        verification = self.grounding.verify(answer, citations)
        unsupported = [item for item in verification["claims"] if not item["supported"]]
        if unsupported:
            warnings.append(f"{len(unsupported)} claim(s) lack sufficient cited evidence")
        confidence = self.grounding.calibrated_confidence(
            evidence_score=evidence_score,
            groundedness=verification["groundedness_score"],
        )

        # 3. Clean answer
        cleaned_answer = self._clean_answer(answer)

        # 4. Check for empty/unhelpful answer
        if len(cleaned_answer.strip()) < 10:
            warnings.append("Answer is very short, may lack detail")
            confidence = min(confidence, 0.2)
        if evidence_score < settings.RETRIEVAL_MIN_EVIDENCE_SCORE:
            warnings.append("Retrieved evidence is insufficient for a reliable answer")
            confidence = min(confidence, settings.NO_ANSWER_CONFIDENCE_THRESHOLD - 0.01)

        return {
            "answer": cleaned_answer,
            "confidence_score": round(confidence, 3),
            "citation_valid": citation_valid,
            "warnings": warnings,
            "groundedness_score": verification["groundedness_score"],
            "claim_support": verification["claims"],
        }

    def _validate_citations(
        self, answer: str, citations: List[Dict]
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

        is_valid = len(invalid) == 0
        return is_valid, warnings

    def _calculate_confidence(
        self,
        answer: str,
        semantic_chunks_used: int,
        graph_nodes_used: int,
        retrieval_mode: str,
        citation_count: int,
    ) -> float:
        """
        Calculate confidence score (0.0 - 1.0) based on:
        - Number of retrieval sources
        - Retrieval mode richness
        - Answer length
        - Citation presence
        """
        score = 0.3  # Base score

        # Source coverage (more sources = higher confidence)
        if semantic_chunks_used >= 3:
            score += 0.15
        elif semantic_chunks_used >= 1:
            score += 0.08

        if graph_nodes_used >= 2:
            score += 0.15
        elif graph_nodes_used >= 1:
            score += 0.08

        # Retrieval mode bonus
        mode_bonus = {
            "hybrid": 0.15,
            "local": 0.12,
            "global": 0.12,
            "semantic": 0.05,
            "graph": 0.05,
        }
        score += mode_bonus.get(retrieval_mode, 0)

        # Citation bonus
        if citation_count >= 3:
            score += 0.1
        elif citation_count >= 1:
            score += 0.05

        # Answer length bonus (longer = more detailed)
        if len(answer) > 500:
            score += 0.1
        elif len(answer) > 200:
            score += 0.05

        # Hedging penalty — phrases indicating uncertainty
        hedging_phrases = [
            "i don't know", "không biết", "không có thông tin",
            "context does not contain", "cannot determine",
            "không đủ thông tin", "i'm not sure",
        ]
        for phrase in hedging_phrases:
            if phrase.lower() in answer.lower():
                score -= 0.2
                break

        return max(0.0, min(1.0, score))

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
