"""
Knowledge Graph Builder (v2)
Extracts entities and relationships from text chunks using Ollama LLM,
stores them in Neo4j (with NetworkX fallback), runs entity normalization,
community detection, and report generation.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

import httpx

from app.config import settings
from app.services.neo4j_store import Neo4jStore
from app.services.entity_normalizer import EntityNormalizer
from app.services.community_detector import CommunityDetector
from app.services.community_reporter import CommunityReporter

logger = logging.getLogger(__name__)

# ─── Supported Entity & Relation Types ────────────────────────────────────────

ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "COURSE", "DEPARTMENT",
    "PRODUCT", "LOCATION", "CONCEPT", "PROJECT",
    "TECHNOLOGY", "EVENT", "DOCUMENT",
]

RELATION_TYPES = [
    "WORKS_AT", "BELONGS_TO", "HAS_PREREQUISITE", "MENTIONS",
    "RELATED_TO", "LOCATED_IN", "MANAGES", "USES",
    "CREATED_BY", "DEPENDS_ON", "HAS_RISK", "PART_OF",
]

EXTRACTION_PROMPT = """You are an expert information extractor. Analyze the following text and extract entities and relationships.

Entity types: {entity_types}
Relationship types: {relation_types}

Rules:
- Only extract clearly stated entities and relationships
- Use UPPERCASE for entity labels (normalize names)
- Include a short verbatim evidence span, confidence from 0 to 1, and temporal bounds when stated
- Return ONLY valid JSON, no other text

Text:
\"\"\"
{text}
\"\"\"

Return JSON in exactly this format:
{{
  "entities": [
    {{"id": "unique_id", "label": "ENTITY NAME", "type": "ENTITY_TYPE"}}
  ],
  "relationships": [
    {{"source": "ENTITY NAME 1", "target": "ENTITY NAME 2", "relation": "RELATION_TYPE", "description": "brief description", "evidence": "supporting text span", "confidence": 0.0, "valid_from": null, "valid_to": null}}
  ]
}}"""


def build_extraction_payload(prompt: str) -> Dict[str, Any]:
    """Build a deterministic Ollama request constrained to a JSON object."""
    return {
        "model": settings.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 512},
    }


class GraphBuilderService:
    """Build and manage knowledge graph with Neo4j, entity normalization, and community detection."""

    def __init__(self):
        # Neo4j store
        self.neo4j = Neo4jStore()

        # Entity normalization
        self.normalizer = EntityNormalizer()

        # Community detection
        self.community_detector = CommunityDetector(
            resolution=settings.COMMUNITY_RESOLUTION,
            min_community_size=settings.COMMUNITY_MIN_SIZE,
        )

        # Community report generator
        self.community_reporter = CommunityReporter()

        logger.info(
            f"GraphBuilder initialized — Neo4j: {'connected' if self.neo4j.is_connected else 'disconnected'}"
        )

    # ─── Extraction ───────────────────────────────────────────────────────────

    async def process_chunks(self, chunks: List[Dict[str, Any]], progress_callback=None) -> int:
        """
        Extract entities/relationships from chunks, normalize, store in Neo4j.
        Returns count of entities added.
        """
        import asyncio

        total_added = 0
        semaphore = asyncio.Semaphore(3)

        async def process_single_chunk(chunk: Dict[str, Any]) -> Optional[Tuple[Dict, str, str]]:
            async with semaphore:
                try:
                    result = await self._extract_from_text(chunk["content"])
                    if result:
                        return result, chunk["document_id"], chunk.get("id", "")
                except Exception as e:
                    logger.error(f"Graph extraction failed for chunk {chunk.get('id')}: {e}")
            return None

        tasks = [process_single_chunk(c) for c in chunks]
        processed = 0
        for completed_task in asyncio.as_completed(tasks):
            res = await completed_task
            if res:
                extraction, doc_id, chunk_id = res
                added = self._merge_into_graph(extraction, doc_id, chunk_id)
                total_added += added
            processed += 1
            if progress_callback and (processed % 5 == 0 or processed == len(chunks)):
                callback_result = progress_callback(processed, len(chunks))
                if asyncio.iscoroutine(callback_result):
                    await callback_result

        if total_added > 0:
            logger.info(f"Graph updated: +{total_added} entities")

            # Run community detection after significant updates
            if total_added >= 3:
                await self.run_community_pipeline()

        return total_added

    async def _extract_from_text(self, text: str) -> Optional[Dict]:
        """Call Ollama LLM to extract entities and relationships from text."""
        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            text=text[:1500],
        )

        payload = build_extraction_payload(prompt)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw)
            if not match:
                logger.warning("No JSON object found in extraction response")
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                return None

        if isinstance(data, dict):
            if "entities" in data and "relationships" in data:
                return data
        logger.warning("Extraction JSON is missing entities or relationships arrays")
        return None

    # ─── Graph Operations ─────────────────────────────────────────────────────

    def _merge_into_graph(self, extraction: Dict, document_id: str, chunk_id: str = "") -> int:
        """Normalize and merge extracted entities/relationships into Neo4j."""
        entities_added = 0
        raw_entities = extraction.get("entities", [])
        raw_relationships = extraction.get("relationships", [])

        # Normalize entities
        existing_labels = []  # Could query Neo4j for existing labels
        normalized_entities = self.normalizer.normalize_entities(raw_entities, existing_labels)
        label_map = self.normalizer.get_label_map(normalized_entities)
        # Ollama may reference relationships by entity `id` even when the prompt
        # asks for names. Resolve both shapes to the normalized entity label.
        normalized_by_original = {
            str(entity.get("original_label", "")).upper().strip(): entity["label"]
            for entity in normalized_entities
        }
        for raw_entity in raw_entities:
            entity_id = str(raw_entity.get("id", "")).upper().strip()
            original_label = str(raw_entity.get("label", "")).upper().strip()
            normalized_label = normalized_by_original.get(original_label)
            if entity_id and normalized_label:
                label_map[entity_id] = normalized_label

        # Normalize relationships
        normalized_rels = self.normalizer.normalize_relationships(raw_relationships, label_map)

        # Upsert entities into Neo4j
        node_id_map = {}
        for ent in normalized_entities:
            label = ent["label"]
            etype = ent["type"]
            node_id = f"{etype}::{label}"
            node_id_map[label] = node_id

            properties = {}
            if chunk_id:
                properties["source_chunk_ids"] = [chunk_id]

            is_new = self.neo4j.upsert_entity(
                node_id=node_id,
                label=label,
                entity_type=etype,
                document_id=document_id,
                properties=properties,
            )
            if is_new:
                entities_added += 1

        # Upsert relationships
        for rel in normalized_rels:
            src_id = node_id_map.get(rel["source"])
            tgt_id = node_id_map.get(rel["target"])

            if src_id and tgt_id:
                try:
                    confidence = float(rel.get("confidence", 0.5) or 0.5)
                except (TypeError, ValueError):
                    confidence = 0.5
                self.neo4j.upsert_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=rel["relation"],
                    document_id=document_id,
                    description=rel.get("description", ""),
                    chunk_id=chunk_id,
                    evidence=rel.get("evidence", ""),
                    confidence=confidence,
                    valid_from=rel.get("valid_from"),
                    valid_to=rel.get("valid_to"),
                )

        return entities_added

    # ─── Community Pipeline ───────────────────────────────────────────────────

    async def run_community_pipeline(self):
        """Run full community detection + report generation pipeline."""
        logger.info("Starting community detection pipeline...")

        try:
            # 1. Export graph data for igraph
            graph_data = self.neo4j.get_all_entities_for_community_detection()
            nodes = graph_data.get("nodes", [])
            edges = graph_data.get("edges", [])

            if len(nodes) < 3:
                logger.info("Too few nodes for community detection, skipping")
                return

            # 2. Run Leiden community detection
            result = self.community_detector.detect(nodes, edges)
            levels = result.get("levels", [])

            if not levels:
                logger.warning("No communities detected")
                return

            # 3. Assign communities to entities in Neo4j
            for level_data in levels:
                for community in level_data.get("communities", []):
                    for member in community.get("members", []):
                        self.neo4j.set_entity_community(
                            node_id=member["node_id"],
                            community_id=community["community_id"],
                            level=level_data["level"],
                        )

            # 4. Generate community reports (use level 1 = medium resolution)
            target_level = levels[1] if len(levels) > 1 else levels[0]
            communities = target_level.get("communities", [])

            reports = await self.community_reporter.generate_reports_batch(communities)

            # 5. Store reports in Neo4j
            for report in reports:
                self.neo4j.upsert_community_report(
                    community_id=report["community_id"],
                    level=report.get("level", 0),
                    title=report.get("title", ""),
                    summary=report.get("summary", ""),
                    key_findings=report.get("key_findings", []),
                    main_entities=report.get("main_entities", []),
                    importance_score=report.get("importance_score", 0.5),
                )

            logger.info(
                f"Community pipeline complete: {len(levels)} levels, "
                f"{sum(len(level['communities']) for level in levels)} communities, "
                f"{len(reports)} reports generated"
            )

        except Exception as e:
            logger.error(f"Community pipeline failed: {e}", exc_info=True)

    # ─── Query (delegated to Neo4jStore) ──────────────────────────────────────

    def get_graph_data(
        self,
        document_ids: Optional[List[str]] = None,
        max_nodes: int = 200,
    ) -> Dict:
        """Return nodes and edges for visualization."""
        return self.neo4j.get_graph_data(document_ids=document_ids, max_nodes=max_nodes)

    def find_entities_in_text(self, text: str) -> List[str]:
        """Find known graph entities mentioned in a query text."""
        return self.neo4j.find_entities_by_label(text)

    def get_related_context(self, entity_ids: List[str], depth: int = 1) -> List[Dict]:
        """Return textual context strings for graph neighborhood."""
        return self.neo4j.get_related_context(entity_ids, depth=depth)

    def get_community_reports(self, level: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """Get community reports."""
        return self.neo4j.get_community_reports(level=level, limit=limit)

    @property
    def stats(self) -> Dict:
        return self.neo4j.stats

    def delete_document_entities(self, document_id: str):
        """Remove all entities that belong only to this document."""
        self.neo4j.delete_document_entities(document_id)
