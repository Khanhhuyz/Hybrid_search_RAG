"""
Context Builder
Assembles retrieval results into a structured prompt context with:
- Token budget management
- Deduplication
- Relevance-based ranking
- XML-tagged source formatting
- Anti-prompt-injection via HTML escaping
"""
import logging
from html import escape
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build structured context for LLM prompts from retrieval results."""

    def __init__(self, max_context_tokens: int = None):
        self.max_context_tokens = max_context_tokens or settings.MAX_CONTEXT_TOKENS
        # Approximate chars per token (rough estimate for multilingual)
        self._chars_per_token = 3.5

    @property
    def max_chars(self) -> int:
        return int(self.max_context_tokens * self._chars_per_token)

    def build(
        self,
        semantic_results: List[Dict],
        graph_context: List[Dict],
        community_context: Optional[List[Dict]] = None,
        question: str = "",
    ) -> Dict[str, str]:
        """
        Build formatted context sections with token budget management.

        Returns:
            {
                "semantic_context": str,
                "graph_context": str,
                "community_context": str,
                "total_chars_used": int,
            }
        """
        budget = self.max_chars
        result = {}

        # 1. Graph context — entities and relationships (allocate 25% budget)
        graph_budget = int(budget * 0.25)
        graph_text = self._format_graph_context(graph_context, graph_budget)
        result["graph_context"] = graph_text
        budget -= len(graph_text)

        # 2. Community context — community reports (allocate 25% budget)
        community_budget = int(budget * 0.30)
        community_text = ""
        if community_context:
            community_text = self._format_community_context(
                community_context, community_budget
            )
        result["community_context"] = community_text
        budget -= len(community_text)

        # 3. Semantic context — document chunks (use remaining budget)
        semantic_text = self._format_semantic_context(semantic_results, budget)
        result["semantic_context"] = semantic_text

        total_used = len(graph_text) + len(community_text) + len(semantic_text)
        result["total_chars_used"] = total_used

        logger.debug(
            f"Context built: {total_used} chars "
            f"(semantic={len(semantic_text)}, graph={len(graph_text)}, "
            f"community={len(community_text)})"
        )
        return result

    # ─── Semantic Context ────────────────────────────────────────────────────

    def _format_semantic_context(
        self, results: List[Dict], budget: int
    ) -> str:
        """Format document chunks as XML-tagged sources with anti-injection."""
        if not results:
            return "No relevant document chunks found."

        parts = []
        used = 0

        for i, r in enumerate(results):
            filename = escape(r.get("document_filename", "Unknown"), quote=True)
            page = r.get("page_number", "?")
            text = r.get("content", "")

            header = (
                f'<source id="S{i+1}" filename="{filename}" page="{page}">\n'
            )
            footer = "\n</source>"
            remaining = budget - used
            available = remaining - len(header) - len(footer) - 2

            if available <= 50:
                break

            # Anti-injection: escape content
            safe_text = escape(text[:available])
            block = f"{header}{safe_text}{footer}"
            parts.append(block)
            used += len(block)

            if len(safe_text) < len(escape(text)):
                break

        return "\n\n".join(parts) if parts else "No relevant document chunks found."

    # ─── Graph Context ───────────────────────────────────────────────────────

    def _format_graph_context(
        self, items: List[Dict], budget: int
    ) -> str:
        """Format graph relationships as structured text."""
        if not items:
            return ""

        parts = []
        used = 0

        parts.append("[Knowledge Graph Relationships]")
        used += len(parts[0])

        for item in items:
            text = f"• {escape(item.get('text', ''))}"
            if used + len(text) + 1 > budget:
                break
            parts.append(text)
            used += len(text) + 1

        return "\n".join(parts) if len(parts) > 1 else ""

    # ─── Community Context ───────────────────────────────────────────────────

    def _format_community_context(
        self, reports: List[Dict], budget: int
    ) -> str:
        """Format community reports as structured summaries."""
        if not reports:
            return ""

        parts = []
        used = 0

        parts.append("[Community Insights]")
        used += len(parts[0])

        for i, report in enumerate(reports):
            title = escape(report.get("title", f"Community {i+1}"))
            summary = escape(report.get("summary", ""))
            findings = report.get("key_findings", [])

            block_parts = [f"\n[C{i+1}] {title}"]
            block_parts.append(f"  Summary: {summary}")

            if findings:
                for f in findings[:3]:
                    block_parts.append(f"  • {escape(f)}")

            block = "\n".join(block_parts)

            if used + len(block) > budget:
                break

            parts.append(block)
            used += len(block)

        return "\n".join(parts) if len(parts) > 1 else ""

    # ─── Deduplication ───────────────────────────────────────────────────────

    @staticmethod
    def dedup_results(results: List[Dict], key: str = "id") -> List[Dict]:
        """Remove duplicate results by key."""
        seen = set()
        deduped = []
        for r in results:
            k = r.get(key)
            if k and k not in seen:
                seen.add(k)
                deduped.append(r)
        return deduped
