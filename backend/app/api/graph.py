"""
Graph API (v2)
Endpoints for knowledge graph visualization, entity queries,
community reports, and relationship queries — backed by Neo4j.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from app.schemas import (
    GraphResponse, EntityQueryRequest, GraphNode, GraphEdge,
    CommunityReport, CommunityListResponse,
)
from app.services.graph_builder import GraphBuilderService
from app.dependencies import get_graph_builder

router = APIRouter(prefix="/graph", tags=["Graph"])
logger = logging.getLogger(__name__)


# ─── Full Graph Visualization ─────────────────────────────────────────────────

@router.get("/visualize", response_model=GraphResponse)
async def get_graph(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    max_nodes: int = Query(200, ge=10, le=500),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """Return nodes and edges for graph visualization."""
    doc_filter = [document_id] if document_id else None
    data = graph_builder.get_graph_data(
        document_ids=doc_filter,
        max_nodes=max_nodes,
    )

    nodes = []
    for n in data.get("nodes", []):
        try:
            nodes.append(GraphNode(**n))
        except Exception:
            continue

    edges = []
    for e in data.get("edges", []):
        try:
            edges.append(GraphEdge(**e))
        except Exception:
            continue

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        total_nodes=data.get("total_nodes", len(nodes)),
        total_edges=data.get("total_edges", len(edges)),
    )


# ─── Entity Query ─────────────────────────────────────────────────────────────

@router.post("/entity/query", response_model=GraphResponse)
async def query_entity(
    request: EntityQueryRequest,
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """Get a subgraph centered on a specific entity with neighborhood expansion."""
    # Find entities matching the name
    entity_ids = graph_builder.find_entities_in_text(request.entity_name)
    if not entity_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{request.entity_name}' not found in the knowledge graph",
        )

    data = graph_builder.neo4j.get_entity_neighborhood(entity_ids, depth=request.depth)

    nodes = [GraphNode(**n) for n in data.get("nodes", []) if n.get("id")]
    edges = [GraphEdge(**e) for e in data.get("edges", []) if e.get("source") and e.get("target")]

    return GraphResponse(
        nodes=nodes,
        edges=edges,
        total_nodes=data.get("total_nodes", len(nodes)),
        total_edges=data.get("total_edges", len(edges)),
    )


# ─── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_graph_stats(graph_builder: GraphBuilderService = Depends(get_graph_builder)):
    """Return statistics about the knowledge graph."""
    stats = graph_builder.stats
    return stats


# ─── Community Reports ────────────────────────────────────────────────────────

@router.get("/communities", response_model=CommunityListResponse)
async def list_communities(
    level: Optional[int] = Query(None, description="Filter by community level"),
    limit: int = Query(50, ge=1, le=200),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """List community reports from the knowledge graph."""
    reports = graph_builder.get_community_reports(level=level, limit=limit)
    community_reports = [
        CommunityReport(
            community_id=r.get("community_id", 0),
            level=r.get("level", 0),
            title=r.get("title", ""),
            summary=r.get("summary", ""),
            key_findings=r.get("key_findings", []),
            main_entities=r.get("main_entities", []),
            importance_score=r.get("importance_score", 0.0),
        )
        for r in reports
    ]
    return CommunityListResponse(
        communities=community_reports,
        total=len(community_reports),
    )


@router.post("/communities/detect")
async def run_community_detection(
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """Manually trigger community detection and report generation."""
    try:
        await graph_builder.run_community_pipeline()
        reports = graph_builder.get_community_reports()
        return {
            "status": "completed",
            "communities_detected": len(reports),
            "message": f"Detected and generated reports for {len(reports)} communities",
        }
    except Exception as e:
        logger.error(f"Community detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── All Entities ─────────────────────────────────────────────────────────────

@router.get("/entities")
async def list_entities(
    entity_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """List all entities in the knowledge graph (via Neo4j)."""
    try:
        from neo4j import GraphDatabase
        query_str = "MATCH (e:Entity) "
        params = {"limit": limit}
        if entity_type:
            query_str += "WHERE e.entity_type = $etype "
            params["etype"] = entity_type.upper()
        query_str += "RETURN e ORDER BY COUNT { (e)-[:RELATES_TO]-() } DESC LIMIT $limit"

        entities = []
        with graph_builder.neo4j._driver.session(database="neo4j") as session:
            result = session.run(query_str, params)
            for record in result:
                e = record["e"]
                entities.append({
                    "id": e.get("node_id"),
                    "label": e.get("label"),
                    "type": e.get("entity_type"),
                    "document_count": len(e.get("document_ids", [])),
                    "community_id": e.get("community_id"),
                })
        return {"entities": entities, "total": len(entities)}
    except Exception as e:
        logger.error(f"Failed to list entities: {e}")
        return {"entities": [], "total": 0}


# ─── Relationships ────────────────────────────────────────────────────────────

@router.get("/relationships")
async def list_relationships(
    relation_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """List relationships in the knowledge graph (via Neo4j)."""
    try:
        query_str = """
        MATCH (src:Entity)-[r:RELATES_TO]->(tgt:Entity)
        """
        params = {"limit": limit}
        if relation_type:
            query_str += "WHERE r.relation_type = $rtype "
            params["rtype"] = relation_type.upper()
        query_str += """
        RETURN src.label AS source, tgt.label AS target,
               r.relation_type AS relation, r.weight AS weight,
               coalesce(r.description, '') AS description
        LIMIT $limit
        """

        rels = []
        with graph_builder.neo4j._driver.session(database="neo4j") as session:
            result = session.run(query_str, params)
            for record in result:
                rels.append({
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "weight": record["weight"],
                    "description": record["description"],
                })
        return {"relationships": rels, "total": len(rels)}
    except Exception as e:
        logger.error(f"Failed to list relationships: {e}")
        return {"relationships": [], "total": 0}
