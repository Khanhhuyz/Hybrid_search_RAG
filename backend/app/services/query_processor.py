"""
Query Processor
Handles query preprocessing: intent analysis, entity extraction,
query classification (local/global/hybrid), and query rewriting.
"""
import logging
import json
import re
from typing import Dict, List, Optional
from enum import Enum

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    LOCAL = "local"      # Specific entity-focused questions
    GLOBAL = "global"    # Broad, summarization questions
    HYBRID = "hybrid"    # Mix of both
    BASIC = "basic"      # Simple vector search


# Keywords that suggest global queries
GLOBAL_KEYWORDS = {
    "tổng quan", "toàn bộ", "tất cả", "chung", "phổ biến", "xu hướng",
    "overall", "summary", "all", "common", "trends", "general",
    "main themes", "chủ đề chính", "vấn đề chung", "rủi ro chung",
    "most important", "quan trọng nhất", "key takeaways",
}

# Keywords that suggest local queries
LOCAL_KEYWORDS = {
    "ai", "who", "người nào", "liên quan đến", "quản lý", "manages",
    "thuộc về", "belongs to", "sử dụng", "uses", "works at", "located",
    "cụ thể", "specific", "chi tiết", "detail",
}

CLASSIFICATION_PROMPT = """Analyze this question and classify it.

Question: {question}

Classify as:
- "local" if the question asks about specific entities, people, projects, or their relationships
- "global" if the question asks for a broad overview, summary, or general patterns across all data
- "hybrid" if it combines both specific and broad aspects

Also extract any entity names mentioned in the question.

Return ONLY valid JSON:
{{
  "query_type": "local" or "global" or "hybrid",
  "entities": ["entity1", "entity2"],
  "rewritten_query": "improved version of the question for better search"
}}"""


class QueryProcessor:
    """Process and classify incoming queries for optimal retrieval routing."""

    def __init__(self, use_llm_classification: bool = True):
        self.use_llm_classification = use_llm_classification

    async def process(
        self,
        question: str,
        known_entities: Optional[List[str]] = None,
    ) -> Dict:
        """
        Full query processing pipeline.

        Returns:
            {
                "query_type": "local" | "global" | "hybrid",
                "original_question": str,
                "rewritten_query": str,
                "extracted_entities": List[str],
                "matched_graph_entities": List[str],
            }
        """
        # 1. Quick heuristic classification
        heuristic_type = self._heuristic_classify(question)

        # 2. Extract entities from question (match against known graph entities)
        matched_entities = []
        if known_entities:
            matched_entities = self._match_entities(question, known_entities)

        # 3. LLM-based classification for ambiguous cases
        extracted_entities = []
        rewritten_query = question
        final_type = heuristic_type

        if self.use_llm_classification and settings.QUERY_CLASSIFICATION_ENABLED:
            try:
                llm_result = await self._llm_classify(question)
                if llm_result:
                    final_type = llm_result.get("query_type", heuristic_type)
                    extracted_entities = llm_result.get("entities", [])
                    rewritten_query = llm_result.get("rewritten_query", question)
            except Exception as e:
                logger.warning(f"LLM classification failed, using heuristic: {e}")
                final_type = heuristic_type

        # 4. Override: if entities found in graph, lean toward local/hybrid
        if matched_entities and final_type == "global":
            final_type = "hybrid"

        # 5. Override: if no entities and no specific pattern, lean toward basic
        if not matched_entities and not extracted_entities and final_type == "local":
            final_type = "hybrid"

        result = {
            "query_type": final_type,
            "original_question": question,
            "rewritten_query": rewritten_query,
            "extracted_entities": extracted_entities,
            "matched_graph_entities": matched_entities,
        }

        logger.info(
            f"Query classified: type={final_type}, "
            f"graph_entities={len(matched_entities)}, "
            f"extracted={len(extracted_entities)}"
        )
        return result

    # ─── Heuristic Classification ────────────────────────────────────────────

    def _heuristic_classify(self, question: str) -> str:
        """Quick keyword-based classification."""
        q_lower = question.lower()

        global_score = sum(1 for kw in GLOBAL_KEYWORDS if kw in q_lower)
        local_score = sum(1 for kw in LOCAL_KEYWORDS if kw in q_lower)

        # Check for question words that suggest specific queries
        if re.search(r"\b(ai|who|whom|người nào|dự án nào|project)\b", q_lower, re.IGNORECASE):
            local_score += 2

        # Check for aggregation words that suggest global queries
        if re.search(r"\b(bao nhiêu|how many|thống kê|statistics|tổng hợp|summarize)\b", q_lower, re.IGNORECASE):
            global_score += 2

        if global_score > local_score:
            return "global"
        elif local_score > global_score:
            return "local"
        return "hybrid"

    def _match_entities(self, question: str, known_entities: List[str]) -> List[str]:
        """Find known graph entity labels mentioned in the question."""
        q_upper = question.upper()
        matched = []
        for entity in known_entities:
            if len(entity) >= 3 and entity.upper() in q_upper:
                matched.append(entity)
        return matched

    # ─── LLM Classification ──────────────────────────────────────────────────

    async def _llm_classify(self, question: str) -> Optional[Dict]:
        """Use LLM for sophisticated query classification."""
        prompt = CLASSIFICATION_PROMPT.format(question=question)
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None

        try:
            data = json.loads(match.group())
            query_type = data.get("query_type", "hybrid")
            if query_type not in ("local", "global", "hybrid"):
                query_type = "hybrid"
            return {
                "query_type": query_type,
                "entities": data.get("entities", []),
                "rewritten_query": data.get("rewritten_query", question),
            }
        except json.JSONDecodeError:
            return None
