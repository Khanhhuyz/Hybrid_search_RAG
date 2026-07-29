"""
Knowledge Graph Builder
Extracts entities and relationships from text chunks using Ollama LLM,
then stores them in a NetworkX graph persisted as JSON.
"""
import json
import logging
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Supported Entity & Relation Types ────────────────────────────────────────

ENTITY_TYPES = ["PERSON", "ORGANIZATION", "COURSE", "DEPARTMENT", "PRODUCT", "LOCATION", "CONCEPT"]
RELATION_TYPES = ["WORKS_AT", "BELONGS_TO", "HAS_PREREQUISITE", "MENTIONS", "RELATED_TO", "LOCATED_IN"]

EXTRACTION_PROMPT = """You are an expert information extractor. Analyze the following text and extract entities and relationships.

Entity types: {entity_types}
Relationship types: {relation_types}

Rules:
- Only extract clearly stated entities and relationships
- Use UPPERCASE for entity labels (normalize names)
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
    {{"source": "ENTITY NAME 1", "target": "ENTITY NAME 2", "relation": "RELATION_TYPE"}}
  ]
}}"""


class GraphBuilderService:
    """Build and persist a NetworkX knowledge graph from document chunks."""

    def __init__(self):
        self.graph_file = settings.GRAPH_FILE
        self.graph: nx.DiGraph = self._load_graph()

    # ─── Graph Persistence ────────────────────────────────────────────────────

    def _load_graph(self) -> nx.DiGraph:
        if self.graph_file.exists():
            try:
                data = json.loads(self.graph_file.read_text(encoding="utf-8"))
                G = nx.node_link_graph(data, edges="edges")
                logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                return G
            except Exception as e:
                logger.warning(f"Failed to load graph, starting fresh: {e}")
        return nx.DiGraph()

    def save_graph(self):
        data = nx.node_link_data(self.graph, edges="edges")
        self.graph_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ─── Extraction ───────────────────────────────────────────────────────────

    async def process_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Extract entities/relationships from chunks and add to graph in parallel.
        Returns count of entities added.
        """
        import asyncio
        total_added = 0
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent Ollama LLM requests to avoid overloading

        async def process_single_chunk(chunk: Dict[str, Any]) -> Optional[Tuple[Dict, str]]:
            async with semaphore:
                try:
                    result = await self._extract_from_text(chunk["content"])
                    if result:
                        return result, chunk["document_id"]
                except Exception as e:
                    logger.error(f"Graph extraction failed for chunk {chunk.get('id')}: {e}")
            return None

        tasks = [process_single_chunk(c) for c in chunks]
        results = await asyncio.gather(*tasks)

        for res in results:
            if res:
                extraction, doc_id = res
                added = self._merge_into_graph(extraction, doc_id)
                total_added += added

        if total_added > 0:
            self.save_graph()
            logger.info(f"Graph updated: +{total_added} entities. Total: {self.graph.number_of_nodes()} nodes")

        return total_added

    async def _extract_from_text(self, text: str) -> Optional[Dict]:
        """Call Ollama LLM to extract entities and relationships from text."""
        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            text=text[:1500],  # Limit to avoid context overflow
        )

        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512},
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            raw = resp.json().get("response", "")

        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        # Find JSON block
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            if "entities" in data and "relationships" in data:
                return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
        return None

    # ─── Graph Operations ─────────────────────────────────────────────────────

    def _merge_into_graph(self, extraction: Dict, document_id: str) -> int:
        """Merge extracted entities and relationships into the graph."""
        entities_added = 0
        label_to_id = {}

        for ent in extraction.get("entities", []):
            label = str(ent.get("label", "")).upper().strip()
            etype = str(ent.get("type", "CONCEPT")).upper()
            if not label:
                continue

            node_id = f"{etype}::{label}"
            label_to_id[label] = node_id

            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    label=label,
                    type=etype,
                    document_ids=[document_id],
                )
                entities_added += 1
            else:
                # Merge document reference
                existing_docs = self.graph.nodes[node_id].get("document_ids", [])
                if document_id not in existing_docs:
                    existing_docs.append(document_id)
                    self.graph.nodes[node_id]["document_ids"] = existing_docs

        for rel in extraction.get("relationships", []):
            src_label = str(rel.get("source", "")).upper().strip()
            tgt_label = str(rel.get("target", "")).upper().strip()
            relation  = str(rel.get("relation", "RELATED_TO")).upper()

            src_id = label_to_id.get(src_label)
            tgt_id = label_to_id.get(tgt_label)

            if src_id and tgt_id and self.graph.has_node(src_id) and self.graph.has_node(tgt_id):
                if not self.graph.has_edge(src_id, tgt_id):
                    self.graph.add_edge(
                        src_id, tgt_id,
                        relation=relation,
                        weight=1.0,
                        document_ids=[document_id],
                    )
                else:
                    # Increment weight for repeated co-occurrence
                    self.graph[src_id][tgt_id]["weight"] = (
                        self.graph[src_id][tgt_id].get("weight", 1.0) + 0.1
                    )

        return entities_added

    # ─── Query ────────────────────────────────────────────────────────────────

    def get_graph_data(
        self,
        document_ids: Optional[List[str]] = None,
        max_nodes: int = 200,
    ) -> Dict:
        """Return nodes and edges for visualization."""
        G = self.graph

        if document_ids:
            nodes = [
                n for n, d in G.nodes(data=True)
                if any(did in d.get("document_ids", []) for did in document_ids)
            ]
            G = G.subgraph(nodes)

        # Limit for large graphs
        if G.number_of_nodes() > max_nodes:
            # Keep highest-degree nodes
            top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            G = G.subgraph([n for n, _ in top_nodes])

        nodes = [
            {
                "id":           n,
                "label":        d.get("label", n),
                "type":         d.get("type", "CONCEPT"),
                "document_ids": d.get("document_ids", []),
                "properties":   {},
            }
            for n, d in G.nodes(data=True)
        ]
        edges = [
            {
                "source":       u,
                "target":       v,
                "relation":     d.get("relation", "RELATED_TO"),
                "weight":       d.get("weight", 1.0),
                "document_ids": d.get("document_ids", []),
            }
            for u, v, d in G.edges(data=True)
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
        }

    def get_entity_neighborhood(
        self,
        entity_name: str,
        depth: int = 2,
    ) -> Dict:
        """Get all nodes within `depth` hops of a named entity."""
        # Find matching node(s)
        candidates = [
            n for n, d in self.graph.nodes(data=True)
            if entity_name.upper() in d.get("label", "").upper()
        ]
        if not candidates:
            return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

        # BFS to find neighborhood
        neighborhood = set()
        for seed in candidates:
            try:
                neighbors = nx.single_source_shortest_path_length(self.graph, seed, cutoff=depth)
                neighborhood.update(neighbors.keys())
                # Also check reverse direction
                rev = self.graph.reverse()
                rev_neighbors = nx.single_source_shortest_path_length(rev, seed, cutoff=depth)
                neighborhood.update(rev_neighbors.keys())
            except Exception:
                neighborhood.add(seed)

        return self.get_graph_data(max_nodes=100)

    def find_entities_in_text(self, text: str) -> List[str]:
        """Find known graph entities mentioned in a query text."""
        text_upper = text.upper()
        found = []
        for node_id, data in self.graph.nodes(data=True):
            label = data.get("label", "")
            if label and len(label.strip()) >= 3 and label.upper() in text_upper:
                found.append(node_id)
        return found

    def get_related_context(self, entity_ids: List[str], depth: int = 1) -> List[Dict]:
        """Return textual context strings for graph neighborhood."""
        context = []
        for eid in entity_ids:
            if not self.graph.has_node(eid):
                continue
            node_data = self.graph.nodes[eid]
            label = node_data.get("label", eid)
            etype = node_data.get("type", "CONCEPT")
            context.append({"text": f"{label} is a {etype}", "source": "graph"})

            for _, tgt, edge_data in self.graph.out_edges(eid, data=True):
                tgt_label = self.graph.nodes[tgt].get("label", tgt)
                relation  = edge_data.get("relation", "RELATED_TO")
                context.append({"text": f"{label} {relation.replace('_', ' ')} {tgt_label}", "source": "graph"})

            for src, _, edge_data in self.graph.in_edges(eid, data=True):
                src_label = self.graph.nodes[src].get("label", src)
                relation  = edge_data.get("relation", "RELATED_TO")
                context.append({"text": f"{src_label} {relation.replace('_', ' ')} {label}", "source": "graph"})

        return context[:20]  # Limit context

    @property
    def stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_types": self._count_types(),
        }

    def _count_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            t = d.get("type", "UNKNOWN")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def delete_document_entities(self, document_id: str):
        """Remove all entities that belong only to this document."""
        to_remove = []
        for node_id, data in self.graph.nodes(data=True):
            doc_ids = data.get("document_ids", [])
            if document_id in doc_ids:
                doc_ids.remove(document_id)
                if not doc_ids:
                    to_remove.append(node_id)
                else:
                    self.graph.nodes[node_id]["document_ids"] = doc_ids

        self.graph.remove_nodes_from(to_remove)
        self.save_graph()
        logger.info(f"Removed {len(to_remove)} orphaned entities for document {document_id}")
