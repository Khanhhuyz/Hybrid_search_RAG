"""
Local Search Service
For entity-focused questions: entity linking → subgraph extraction →
path finding → source chunk retrieval → scoring.
"""
import logging
from typing import Dict, List, Any, Optional

from app.services.neo4j_store import Neo4jStore
from app.services.embedder import EmbedderService
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class LocalSearch:
    """
    Local Search pipeline for entity-specific questions.
    Combines graph traversal with vector retrieval for precise answers.
    """

    def __init__(
        self,
        neo4j_store: Neo4jStore,
        embedder: EmbedderService,
        vector_store: VectorStoreService,
    ):
        self.neo4j = neo4j_store
        self.embedder = embedder
        self.vector_store = vector_store

    async def search(
        self,
        question: str,
        query_vector: list,
        entity_ids: List[str],
        top_k: int = 5,
        graph_depth: int = 2,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute local search pipeline.

        1. Get entity neighborhood subgraph
        2. Get related context text
        3. Get community reports for matched entities
        4. Get relevant chunks via vector search
        5. Combine and score

        Returns:
            {
                "graph_context": [...],
                "semantic_results": [...],
                "community_context": [...],
                "subgraph": {...},
                "retrieval_mode": "local"
            }
        """
        # 1. Graph traversal — multi-hop neighborhood
        subgraph = {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}
        graph_context = []

        if entity_ids and self.neo4j.is_connected:
            subgraph = self.neo4j.get_entity_neighborhood(entity_ids, depth=graph_depth)
            graph_context = self.neo4j.get_related_context(entity_ids, depth=graph_depth)

        # 2. Community context — get reports for communities of matched entities
        community_context = []
        if entity_ids and self.neo4j.is_connected:
            community_context = self._get_entity_community_reports(entity_ids)

        # 3. Semantic search — vector similarity
        semantic_results = await self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Fetch extra for RRF
            document_ids=document_ids,
        )

        # 4. Source chunk retrieval — find chunks that mention the entities
        entity_chunks = []
        if entity_ids and self.neo4j.is_connected:
            entity_chunks = await self._find_entity_source_chunks(entity_ids, document_ids)

        # 5. Merge and deduplicate
        all_chunks = self._merge_and_dedup(semantic_results, entity_chunks)

        return {
            "graph_context": graph_context,
            "semantic_results": all_chunks[:top_k * 2],
            "community_context": community_context,
            "subgraph": subgraph,
            "entity_ids": entity_ids,
            "retrieval_mode": "local",
        }

    def _get_entity_community_reports(self, entity_ids: List[str]) -> List[Dict]:
        """Get community reports for the communities that contain the matched entities."""
        reports = self.neo4j.get_community_reports(limit=10)
        if not reports:
            return []

        # Filter reports that mention any of our entities
        relevant = []
        for report in reports:
            main_entities = [e.upper() for e in report.get("main_entities", [])]
            for eid in entity_ids:
                # entity_id format is "TYPE::LABEL", extract label
                label = eid.split("::")[-1].upper() if "::" in eid else eid.upper()
                if any(label in me for me in main_entities):
                    relevant.append(report)
                    break

        return relevant[:5]

    async def _find_entity_source_chunks(
        self,
        entity_ids: List[str],
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Find chunks that are source documents for the given entities."""
        # Get document_ids from entities
        entity_doc_ids = set()
        if self.neo4j.is_connected:
            for eid in entity_ids:
                # Query entity's document_ids
                try:
                    from neo4j import GraphDatabase
                    with self.neo4j._driver.session(database="neo4j") as session:
                        result = session.run(
                            "MATCH (e:Entity {node_id: $eid}) RETURN e.document_ids AS docs",
                            {"eid": eid},
                        )
                        record = result.single()
                        if record and record["docs"]:
                            entity_doc_ids.update(record["docs"])
                except Exception:
                    pass

        # Filter by requested document_ids if provided
        if document_ids:
            entity_doc_ids = entity_doc_ids.intersection(set(document_ids))

        if not entity_doc_ids:
            return []

        # Search within those documents
        # For simplicity, we use a generic query embedding
        try:
            labels = []
            for eid in entity_ids[:3]:
                label = eid.split("::")[-1] if "::" in eid else eid
                labels.append(label)
            entity_query = " ".join(labels)
            query_vector = await self.embedder.embed_query(entity_query)
            results = await self.vector_store.similarity_search(
                query_vector=query_vector,
                top_k=5,
                document_ids=list(entity_doc_ids),
            )
            return results
        except Exception as e:
            logger.warning(f"Entity source chunk search failed: {e}")
            return []

    def _merge_and_dedup(
        self, semantic: List[Dict], entity_chunks: List[Dict]
    ) -> List[Dict]:
        """Merge two lists of chunks, deduplicating by chunk id."""
        seen_ids = set()
        merged = []

        for chunk in semantic:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

        for chunk in entity_chunks:
            cid = chunk.get("id")
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(chunk)

        return merged
