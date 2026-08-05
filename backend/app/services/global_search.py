"""
Global Search Service
For broad/summary questions: Map-Reduce over community reports.
Inspired by Microsoft GraphRAG global search.
"""
import json
import logging
import re
from typing import Dict, List, Any, Optional

import httpx

from app.config import settings
from app.services.neo4j_store import Neo4jStore

logger = logging.getLogger(__name__)

MAP_PROMPT = """You are an expert analyst. Given the following community report and the user's question,
extract any relevant information that could help answer the question.

Question: {question}

Community Report:
Title: {title}
Summary: {summary}
Key Findings: {findings}
Main Entities: {entities}

If this community has relevant information, provide:
1. Key points that help answer the question
2. Supporting evidence from the community

Return ONLY valid JSON:
{{
  "is_relevant": true/false,
  "key_points": ["point1", "point2"],
  "evidence": "brief supporting text",
  "importance": 0.0 to 1.0
}}"""

REDUCE_PROMPT = """You are an expert analyst synthesizing information from multiple data sources.

Question: {question}

The following key points were extracted from different communities in the knowledge base:

{intermediate_results}

Based on ALL the above information:
1. Synthesize a comprehensive answer to the question
2. Organize the answer logically
3. Cite community sources using [C1], [C2], etc.
4. Answer in the SAME language as the question

Provide a thorough, well-structured answer:"""


class GlobalSearch:
    """
    Global Search pipeline using Map-Reduce over community reports.
    For questions requiring broad, cross-document summarization.
    """

    def __init__(self, neo4j_store: Neo4jStore):
        self.neo4j = neo4j_store

    async def search(
        self,
        question: str,
        community_level: Optional[int] = None,
        max_communities: int = None,
    ) -> Dict[str, Any]:
        """
        Execute global search pipeline.

        1. Get community reports (at specified level or all levels)
        2. MAP: LLM analyzes each batch of reports for relevance
        3. Filter & rank intermediate results
        4. REDUCE: LLM synthesizes final answer

        Returns:
            {
                "answer": str,
                "community_citations": [...],
                "intermediate_results": [...],
                "retrieval_mode": "global"
            }
        """
        max_communities = max_communities or settings.GLOBAL_SEARCH_MAX_COMMUNITIES

        # 1. Get community reports
        reports = self.neo4j.get_community_reports(level=community_level, limit=max_communities)

        if not reports:
            return {
                "answer": "",
                "community_citations": [],
                "intermediate_results": [],
                "retrieval_mode": "global",
                "map_results_count": 0,
            }

        logger.info(f"Global search: analyzing {len(reports)} community reports")

        # 2. MAP phase — analyze each batch
        batch_size = settings.GLOBAL_SEARCH_MAP_BATCH_SIZE
        batches = [reports[i:i + batch_size] for i in range(0, len(reports), batch_size)]

        all_map_results = []
        for batch_idx, batch in enumerate(batches):
            batch_results = await self._map_batch(question, batch)
            all_map_results.extend(batch_results)

        # 3. Filter & rank
        relevant_results = [
            r for r in all_map_results
            if r.get("is_relevant", False) and r.get("importance", 0) > 0.2
        ]
        relevant_results.sort(key=lambda x: x.get("importance", 0), reverse=True)

        if not relevant_results:
            return {
                "answer": "No relevant information found across the knowledge base communities.",
                "community_citations": [],
                "intermediate_results": [],
                "retrieval_mode": "global",
                "map_results_count": 0,
            }

        # 4. REDUCE phase — synthesize final answer
        answer, citations = await self._reduce(question, relevant_results[:15])

        return {
            "answer": answer,
            "community_citations": citations,
            "intermediate_results": relevant_results[:15],
            "retrieval_mode": "global",
            "map_results_count": len(relevant_results),
        }

    async def _map_batch(
        self, question: str, reports: List[Dict]
    ) -> List[Dict]:
        """MAP phase: analyze a batch of community reports."""
        import asyncio

        semaphore = asyncio.Semaphore(3)

        async def analyze_single(report: Dict) -> Optional[Dict]:
            async with semaphore:
                return await self._analyze_report(question, report)

        tasks = [analyze_single(r) for r in reports]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = []
        for r in results:
            if isinstance(r, dict):
                valid.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"Map analysis failed: {r}")

        return valid

    async def _analyze_report(self, question: str, report: Dict) -> Optional[Dict]:
        """Analyze a single community report for relevance to the question."""
        findings = "\n".join(f"- {f}" for f in report.get("key_findings", []))
        entities = ", ".join(report.get("main_entities", []))

        prompt = MAP_PROMPT.format(
            question=question,
            title=report.get("title", ""),
            summary=report.get("summary", ""),
            findings=findings or "None",
            entities=entities or "None",
        )

        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 512},
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")

            match = re.search(r"\{[\s\S]*\}", raw)
            if not match:
                return None

            data = json.loads(match.group())
            data["community_id"] = report.get("community_id")
            data["community_title"] = report.get("title", "")
            return data

        except Exception as e:
            logger.warning(f"Report analysis failed: {e}")
            return None

    async def _reduce(
        self, question: str, intermediate_results: List[Dict]
    ) -> tuple:
        """REDUCE phase: synthesize intermediate results into final answer."""
        # Build intermediate results text
        parts = []
        citations = []
        for i, result in enumerate(intermediate_results, 1):
            title = result.get("community_title", f"Community {result.get('community_id', '?')}")
            key_points = "\n".join(f"  - {p}" for p in result.get("key_points", []))
            evidence = result.get("evidence", "")

            parts.append(
                f"[C{i}] Community: {title}\n"
                f"  Key Points:\n{key_points}\n"
                f"  Evidence: {evidence}"
            )

            citations.append({
                "citation_id": f"C{i}",
                "community_id": result.get("community_id"),
                "community_title": title,
                "key_points": result.get("key_points", []),
                "importance": result.get("importance", 0),
            })

        intermediate_text = "\n\n".join(parts)

        prompt = REDUCE_PROMPT.format(
            question=question,
            intermediate_results=intermediate_text,
        )

        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "").strip()
                return answer, citations
        except Exception as e:
            logger.error(f"Reduce phase failed: {e}")
            return "Failed to synthesize answer from community reports.", citations
