"""
Graph API
Endpoints for knowledge graph visualization, entity queries, and relationship queries.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Query

from app.schemas import GraphResponse, EntityQueryRequest, GraphNode, GraphEdge
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
    return GraphResponse(
        nodes=[GraphNode(**n) for n in data["nodes"]],
        edges=[GraphEdge(**e) for e in data["edges"]],
        total_nodes=data["total_nodes"],
        total_edges=data["total_edges"],
    )


# ─── Entity Query ─────────────────────────────────────────────────────────────

@router.post("/entity/query", response_model=GraphResponse)
async def query_entity(
    request: EntityQueryRequest,
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """Get a subgraph centered on a specific entity with neighborhood expansion."""
    data = graph_builder.get_entity_neighborhood(
        entity_name=request.entity_name,
        depth=request.depth,
    )
    if not data["nodes"]:
        raise HTTPException(
            status_code=404,
            detail=f"Entity '{request.entity_name}' not found in the knowledge graph",
        )
    return GraphResponse(
        nodes=[GraphNode(**n) for n in data["nodes"]],
        edges=[GraphEdge(**e) for e in data["edges"]],
        total_nodes=data["total_nodes"],
        total_edges=data["total_edges"],
    )


# ─── Stats ────────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_graph_stats(graph_builder: GraphBuilderService = Depends(get_graph_builder)):
    """Return statistics about the knowledge graph."""
    return graph_builder.stats


# ─── All Entities ─────────────────────────────────────────────────────────────

@router.get("/entities")
async def list_entities(
    entity_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """List all entities in the knowledge graph."""
    entities = []
    for node_id, data in graph_builder.graph.nodes(data=True):
        if entity_type and data.get("type", "").upper() != entity_type.upper():
            continue
        entities.append({
            "id":    node_id,
            "label": data.get("label"),
            "type":  data.get("type"),
            "document_count": len(data.get("document_ids", [])),
        })
        if len(entities) >= limit:
            break
    return {"entities": entities, "total": len(entities)}


# ─── Relationships ────────────────────────────────────────────────────────────

@router.get("/relationships")
async def list_relationships(
    relation_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """List relationships in the knowledge graph."""
    rels = []
    for src, tgt, data in graph_builder.graph.edges(data=True):
        if relation_type and data.get("relation", "").upper() != relation_type.upper():
            continue
        src_label = graph_builder.graph.nodes[src].get("label", src)
        tgt_label = graph_builder.graph.nodes[tgt].get("label", tgt)
        rels.append({
            "source":    src_label,
            "target":    tgt_label,
            "relation":  data.get("relation"),
            "weight":    data.get("weight", 1.0),
        })
        if len(rels) >= limit:
            break
    return {"relationships": rels, "total": len(rels)}
