"""
Neo4j Graph Store
Replaces NetworkX JSON persistence with Neo4j for production-grade
knowledge graph storage, querying, and community detection support.
"""
import logging
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase, Driver

from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Production-grade graph store using Neo4j."""

    def __init__(self):
        self._driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            self._driver.verify_connectivity()
            self._create_indexes()
            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self._driver = None

    def _create_indexes(self):
        """Create indexes for performant lookups."""
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            # Entity indexes
            session.run("CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)")
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
            session.run("CREATE INDEX entity_node_id IF NOT EXISTS FOR (e:Entity) ON (e.node_id)")
            # Community indexes
            session.run("CREATE INDEX community_id IF NOT EXISTS FOR (c:Community) ON (c.community_id)")
            session.run("CREATE INDEX community_level IF NOT EXISTS FOR (c:Community) ON (c.level)")
            # Claim index
            session.run("CREATE INDEX claim_id IF NOT EXISTS FOR (cl:Claim) ON (cl.claim_id)")
            logger.debug("Neo4j indexes ensured")

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    # ─── Entity Operations ───────────────────────────────────────────────────

    def upsert_entity(
        self,
        node_id: str,
        label: str,
        entity_type: str,
        document_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create or update an entity node. Returns True if new node created."""
        query = """
        MERGE (e:Entity {node_id: $node_id})
        ON CREATE SET
            e.label = $label,
            e.entity_type = $entity_type,
            e.document_ids = [$document_id],
            e.created_at = datetime()
        ON MATCH SET
            e.label = CASE WHEN size(e.label) < size($label) THEN $label ELSE e.label END,
            e.document_ids = CASE
                WHEN NOT $document_id IN e.document_ids
                THEN e.document_ids + $document_id
                ELSE e.document_ids
            END,
            e.updated_at = datetime()
        WITH e, NOT exists(e.updated_at) AS is_new
        RETURN is_new
        """
        params = {
            "node_id": node_id,
            "label": label,
            "entity_type": entity_type,
            "document_id": document_id,
        }
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, params)
            record = result.single()

            # Set additional properties if provided
            if properties:
                prop_query = """
                MATCH (e:Entity {node_id: $node_id})
                SET e += $props
                """
                session.run(prop_query, {"node_id": node_id, "props": properties})

            return record["is_new"] if record else False

    def upsert_relationship(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        document_id: str,
        description: str = "",
        weight: float = 1.0,
    ):
        """Create or update a relationship between entities."""
        query = """
        MATCH (src:Entity {node_id: $source_id})
        MATCH (tgt:Entity {node_id: $target_id})
        MERGE (src)-[r:RELATES_TO {relation_type: $relation}]->(tgt)
        ON CREATE SET
            r.weight = $weight,
            r.description = $description,
            r.document_ids = [$document_id],
            r.created_at = datetime()
        ON MATCH SET
            r.weight = r.weight + 0.1,
            r.document_ids = CASE
                WHEN NOT $document_id IN r.document_ids
                THEN r.document_ids + $document_id
                ELSE r.document_ids
            END,
            r.description = CASE WHEN size($description) > size(coalesce(r.description, '')) THEN $description ELSE r.description END
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(query, {
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "document_id": document_id,
                "description": description,
                "weight": weight,
            })

    # ─── Query Operations ────────────────────────────────────────────────────

    def find_entities_by_label(self, text: str, min_length: int = 3) -> List[str]:
        """Find known graph entities mentioned in text (case-insensitive)."""
        query = """
        MATCH (e:Entity)
        WHERE size(e.label) >= $min_length AND toUpper($text) CONTAINS toUpper(e.label)
        RETURN e.node_id AS node_id
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, {"text": text, "min_length": min_length})
            return [r["node_id"] for r in result]

    def get_entity_neighborhood(
        self, entity_ids: List[str], depth: int = 2
    ) -> Dict[str, Any]:
        """Get multi-hop neighborhood subgraph for given entities."""
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {node_id: eid})
        CALL apoc.path.subgraphAll(e, {maxLevel: $depth, relationshipFilter: 'RELATES_TO>'})
        YIELD nodes, relationships
        UNWIND nodes AS n
        WITH COLLECT(DISTINCT n) AS all_nodes,
             COLLECT(DISTINCT relationships) AS all_rels_lists
        UNWIND all_rels_lists AS rels
        UNWIND rels AS r
        WITH all_nodes, COLLECT(DISTINCT r) AS all_rels
        RETURN
            [n IN all_nodes | {
                id: n.node_id,
                label: n.label,
                type: n.entity_type,
                document_ids: coalesce(n.document_ids, []),
                properties: {},
                community_id: n.community_id
            }] AS nodes,
            [r IN all_rels | {
                source: startNode(r).node_id,
                target: endNode(r).node_id,
                relation: r.relation_type,
                weight: r.weight,
                description: coalesce(r.description, ''),
                document_ids: coalesce(r.document_ids, [])
            }] AS edges
        """
        try:
            with self._driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(query, {"entity_ids": entity_ids, "depth": depth})
                record = result.single()
                if record:
                    nodes = record["nodes"]
                    edges = record["edges"]
                    return {
                        "nodes": nodes,
                        "edges": edges,
                        "total_nodes": len(nodes),
                        "total_edges": len(edges),
                    }
        except Exception as e:
            logger.warning(f"APOC path query failed, falling back to basic traversal: {e}")
            return self._basic_neighborhood(entity_ids, depth)

        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    def _basic_neighborhood(self, entity_ids: List[str], depth: int) -> Dict[str, Any]:
        """Fallback neighborhood query without APOC."""
        query = """
        UNWIND $entity_ids AS eid
        MATCH path = (e:Entity {node_id: eid})-[r:RELATES_TO*1..2]->(n:Entity)
        WITH COLLECT(DISTINCT e) + COLLECT(DISTINCT n) AS all_nodes,
             [rel IN COLLECT(DISTINCT r) | rel[0]] AS all_rels
        UNWIND all_nodes AS node
        WITH COLLECT(DISTINCT node) AS nodes, all_rels
        RETURN
            [n IN nodes | {
                id: n.node_id,
                label: n.label,
                type: n.entity_type,
                document_ids: coalesce(n.document_ids, []),
                properties: {},
                community_id: n.community_id
            }] AS nodes,
            size(nodes) AS total_nodes
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, {"entity_ids": entity_ids})
            record = result.single()
            if record:
                return {
                    "nodes": record["nodes"],
                    "edges": [],
                    "total_nodes": record["total_nodes"],
                    "total_edges": 0,
                }
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    def get_related_context(self, entity_ids: List[str], depth: int = 2) -> List[Dict]:
        """Return textual context strings for entities and their relationships."""
        query = """
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {node_id: eid})
        OPTIONAL MATCH (e)-[r:RELATES_TO]->(t:Entity)
        OPTIONAL MATCH (s:Entity)-[r2:RELATES_TO]->(e)
        RETURN
            e.node_id AS entity_id,
            e.label AS label,
            e.entity_type AS type,
            COLLECT(DISTINCT {
                target_label: t.label,
                relation: r.relation_type,
                description: coalesce(r.description, ''),
                direction: 'outgoing'
            }) AS out_rels,
            COLLECT(DISTINCT {
                source_label: s.label,
                relation: r2.relation_type,
                description: coalesce(r2.description, ''),
                direction: 'incoming'
            }) AS in_rels
        """
        context = []
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, {"entity_ids": entity_ids})
            for record in result:
                label = record["label"]
                etype = record["type"]
                context.append({"text": f"{label} is a {etype}", "source": "graph"})

                for rel in record["out_rels"]:
                    if rel["target_label"]:
                        rel_text = rel["relation"].replace("_", " ")
                        desc = f" ({rel['description']})" if rel["description"] else ""
                        context.append({
                            "text": f"{label} {rel_text} {rel['target_label']}{desc}",
                            "source": "graph",
                        })

                for rel in record["in_rels"]:
                    if rel["source_label"]:
                        rel_text = rel["relation"].replace("_", " ")
                        desc = f" ({rel['description']})" if rel["description"] else ""
                        context.append({
                            "text": f"{rel['source_label']} {rel_text} {label}{desc}",
                            "source": "graph",
                        })

        return context[:30]

    # ─── Graph Data for Visualization ────────────────────────────────────────

    def get_graph_data(
        self,
        document_ids: Optional[List[str]] = None,
        max_nodes: int = 200,
    ) -> Dict:
        """Return nodes and edges for frontend visualization."""
        if document_ids:
            query = """
            MATCH (e:Entity)
            WHERE ANY(d IN e.document_ids WHERE d IN $doc_ids)
            WITH e ORDER BY COUNT { (e)-[:RELATES_TO]-() } DESC LIMIT $max_nodes
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(t:Entity)
            WHERE ANY(d IN t.document_ids WHERE d IN $doc_ids)
            RETURN
                COLLECT(DISTINCT {
                    id: e.node_id, label: e.label, type: e.entity_type,
                    document_ids: e.document_ids, properties: {},
                    community_id: e.community_id
                }) AS nodes,
                COLLECT(DISTINCT {
                    source: e.node_id, target: t.node_id,
                    relation: r.relation_type, weight: r.weight,
                    document_ids: coalesce(r.document_ids, [])
                }) AS edges
            """
            params = {"doc_ids": document_ids, "max_nodes": max_nodes}
        else:
            query = """
            MATCH (e:Entity)
            WITH e ORDER BY COUNT { (e)-[:RELATES_TO]-() } DESC LIMIT $max_nodes
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(t:Entity)
            RETURN
                COLLECT(DISTINCT {
                    id: e.node_id, label: e.label, type: e.entity_type,
                    document_ids: e.document_ids, properties: {},
                    community_id: e.community_id
                }) AS nodes,
                COLLECT(DISTINCT {
                    source: e.node_id, target: t.node_id,
                    relation: r.relation_type, weight: r.weight,
                    document_ids: coalesce(r.document_ids, [])
                }) AS edges
            """
            params = {"max_nodes": max_nodes}

        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, params)
            record = result.single()
            if record:
                nodes = [n for n in record["nodes"] if n["id"]]
                edges = [e for e in record["edges"] if e["source"] and e["target"]]
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                }
        return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

    @property
    def stats(self) -> Dict:
        """Return graph statistics."""
        query = """
        MATCH (e:Entity)
        WITH COUNT(e) AS node_count, COLLECT(e.entity_type) AS types
        OPTIONAL MATCH ()-[r:RELATES_TO]->()
        WITH node_count, types, COUNT(r) AS edge_count
        RETURN node_count, edge_count,
               apoc.coll.frequencies(types) AS type_counts
        """
        try:
            with self._driver.session(database=settings.NEO4J_DATABASE) as session:
                result = session.run(query)
                record = result.single()
                if record:
                    type_counts = {}
                    for item in record.get("type_counts", []):
                        type_counts[item["item"]] = item["count"]
                    return {
                        "nodes": record["node_count"],
                        "edges": record["edge_count"],
                        "entity_types": type_counts,
                    }
        except Exception:
            return self._stats_fallback()
        return {"nodes": 0, "edges": 0, "entity_types": {}}

    def _stats_fallback(self) -> Dict:
        """Stats without APOC."""
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            nodes = session.run("MATCH (e:Entity) RETURN COUNT(e) AS c").single()["c"]
            edges = session.run("MATCH ()-[r:RELATES_TO]->() RETURN COUNT(r) AS c").single()["c"]
            types_result = session.run(
                "MATCH (e:Entity) RETURN e.entity_type AS t, COUNT(*) AS c"
            )
            type_counts = {r["t"]: r["c"] for r in types_result}
            return {"nodes": nodes, "edges": edges, "entity_types": type_counts}

    # ─── Community Operations ────────────────────────────────────────────────

    def set_entity_community(self, node_id: str, community_id: int, level: int = 0):
        """Assign an entity to a community."""
        query = """
        MATCH (e:Entity {node_id: $node_id})
        SET e.community_id = $community_id, e.community_level = $level
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(query, {
                "node_id": node_id,
                "community_id": community_id,
                "level": level,
            })

    def upsert_community_report(
        self,
        community_id: int,
        level: int,
        title: str,
        summary: str,
        key_findings: List[str],
        main_entities: List[str],
        importance_score: float,
    ):
        """Store a community report node."""
        query = """
        MERGE (c:Community {community_id: $community_id, level: $level})
        SET c.title = $title,
            c.summary = $summary,
            c.key_findings = $key_findings,
            c.main_entities = $main_entities,
            c.importance_score = $importance_score,
            c.updated_at = datetime()
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(query, {
                "community_id": community_id,
                "level": level,
                "title": title,
                "summary": summary,
                "key_findings": key_findings,
                "main_entities": main_entities,
                "importance_score": importance_score,
            })

    def get_community_reports(
        self,
        level: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve community reports, optionally filtered by level."""
        if level is not None:
            query = """
            MATCH (c:Community {level: $level})
            RETURN c ORDER BY c.importance_score DESC LIMIT $limit
            """
            params = {"level": level, "limit": limit}
        else:
            query = """
            MATCH (c:Community)
            RETURN c ORDER BY c.importance_score DESC LIMIT $limit
            """
            params = {"limit": limit}

        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, params)
            reports = []
            for record in result:
                c = record["c"]
                reports.append({
                    "community_id": c.get("community_id"),
                    "level": c.get("level", 0),
                    "title": c.get("title", ""),
                    "summary": c.get("summary", ""),
                    "key_findings": c.get("key_findings", []),
                    "main_entities": c.get("main_entities", []),
                    "importance_score": c.get("importance_score", 0.0),
                })
            return reports

    # ─── Delete Operations ───────────────────────────────────────────────────

    def delete_document_entities(self, document_id: str):
        """Remove entities and relationships that belong only to this document."""
        # Remove document_id from arrays, then delete orphaned nodes
        query = """
        MATCH (e:Entity)
        WHERE $document_id IN e.document_ids
        SET e.document_ids = [d IN e.document_ids WHERE d <> $document_id]
        WITH e WHERE size(e.document_ids) = 0
        DETACH DELETE e
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, {"document_id": document_id})
            summary = result.consume()
            deleted = summary.counters.nodes_deleted
            logger.info(f"Removed {deleted} orphaned entities for document {document_id}")

    def get_all_entities_for_community_detection(self) -> Dict:
        """Export all entities and relationships as adjacency data for igraph/Leiden."""
        query = """
        MATCH (e:Entity)
        WITH COLLECT({node_id: e.node_id, label: e.label, type: e.entity_type}) AS nodes
        OPTIONAL MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
        WITH nodes, COLLECT({source: src.node_id, target: tgt.node_id, weight: r.weight}) AS edges
        RETURN nodes, edges
        """
        with self._driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query)
            record = result.single()
            if record:
                return {
                    "nodes": record["nodes"],
                    "edges": [e for e in record["edges"] if e["source"] and e["target"]],
                }
        return {"nodes": [], "edges": []}

    def health_check(self) -> str:
        """Check Neo4j connectivity."""
        if not self._driver:
            return "disconnected"
        try:
            self._driver.verify_connectivity()
            return "ok"
        except Exception as e:
            return f"error: {str(e)}"
