"""
Community Report Generator
Uses LLM to generate natural language summaries for each detected community.
Reports include title, summary, key findings, and importance ranking.
"""
import json
import logging
from typing import Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

COMMUNITY_REPORT_PROMPT = """You are an expert analyst. Analyze the following group of related entities and relationships from a knowledge graph.

Generate a structured report about this community/cluster.

Community Members:
{members_text}

Relationships within this community:
{relationships_text}

Rules:
- Use only the supplied relationship evidence. Do not add background knowledge.
- Every key finding must be directly supported by at least one evidence span.
- If the evidence is insufficient, return an empty key_findings list and say so.

Return ONLY valid JSON in exactly this format:
{{
  "title": "A short descriptive title for this community (max 10 words)",
  "summary": "A 2-3 sentence summary of what this community represents and its significance",
  "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
  "importance_score": 0.0 to 1.0 (how important/central is this community)
}}"""


class CommunityReporter:
    """Generate LLM-powered community reports."""

    def __init__(self):
        self._report_cache: Dict[int, Dict] = {}

    async def generate_report(
        self,
        community: Dict,
        relationships: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Generate a report for a single community using LLM.

        Args:
            community: Dict with keys: community_id, members, key_entities
            relationships: List of relationships within this community
        """
        community_id = community["community_id"]

        # Check cache
        if community_id in self._report_cache:
            return self._report_cache[community_id]

        if not relationships:
            return {
                "community_id": community_id,
                "level": community.get("level", 0),
                "title": "Unverified entity cluster",
                "summary": "No verified relationship evidence is available for this cluster.",
                "key_findings": [],
                "main_entities": community.get("key_entities", []),
                "importance_score": 0.0,
            }

        # Build text descriptions of members
        members = community.get("members", [])
        members_text = "\n".join(
            f"- {m['label']} (Type: {m['type']})"
            for m in members[:30]  # Limit to avoid token overflow
        )

        # Build relationships text
        rels_text = "No relationships available."
        if relationships:
            rels_text = "\n".join(
                f"- {r.get('source_label', r.get('source', '?'))} "
                f"→ [{r.get('relation', 'RELATED_TO')}] → "
                f"{r.get('target_label', r.get('target', '?'))}; "
                f"Evidence: {r.get('evidence', '')}"
                for r in relationships[:20]
            )

        prompt = COMMUNITY_REPORT_PROMPT.format(
            members_text=members_text,
            relationships_text=rels_text,
        )

        try:
            report = await self._call_llm(prompt)
            report["community_id"] = community_id
            report["level"] = community.get("level", 0)
            report["main_entities"] = community.get("key_entities", [])

            # Cache the result
            self._report_cache[community_id] = report
            return report

        except Exception as e:
            logger.error(f"Failed to generate report for community {community_id}: {e}")
            # Fallback report
            return {
                "community_id": community_id,
                "level": community.get("level", 0),
                "title": f"Community {community_id}",
                "summary": f"A cluster of {len(members)} related entities including "
                           f"{', '.join(community.get('key_entities', [])[:3])}.",
                "key_findings": [],
                "main_entities": community.get("key_entities", []),
                "importance_score": 0.5,
            }

    async def generate_reports_batch(
        self,
        communities: List[Dict],
        relationships_by_community: Optional[Dict[int, List[Dict]]] = None,
    ) -> List[Dict]:
        """Generate reports for multiple communities."""
        import asyncio

        if relationships_by_community is None:
            relationships_by_community = {}

        semaphore = asyncio.Semaphore(3)

        async def gen_single(community: Dict) -> Dict:
            async with semaphore:
                rels = relationships_by_community.get(community["community_id"], [])
                return await self.generate_report(community, rels)

        tasks = [gen_single(c) for c in communities]
        reports = await asyncio.gather(*tasks, return_exceptions=True)

        valid_reports = []
        for r in reports:
            if isinstance(r, Exception):
                logger.error(f"Report generation error: {r}")
            elif isinstance(r, dict):
                valid_reports.append(r)

        logger.info(f"Generated {len(valid_reports)}/{len(communities)} community reports")
        return valid_reports

    async def _call_llm(self, prompt: str) -> Dict:
        """Call Ollama LLM and parse JSON response."""
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": settings.COMMUNITY_MAX_REPORT_TOKENS},
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        # Parse JSON from response
        import re
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise ValueError(f"No JSON found in LLM response: {raw[:200]}")

        data = json.loads(match.group())
        return {
            "title": data.get("title", "Unnamed Community"),
            "summary": data.get("summary", ""),
            "key_findings": data.get("key_findings", []),
            "importance_score": float(data.get("importance_score", 0.5)),
        }

    def clear_cache(self):
        """Clear report cache."""
        self._report_cache.clear()
