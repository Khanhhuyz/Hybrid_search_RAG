"""
Search API (v2)
Endpoints for semantic search, graph search, local/global/hybrid GraphRAG chat,
and monitoring stats.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    ChunkResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    GraphContext,
    MonitoringStats,
    QueryLogEntry,
)
from app.services.rag_pipeline import RAGPipeline
from app.services.monitor import Monitor
from app.dependencies import get_rag_pipeline, get_monitor

router = APIRouter(tags=["Search"])
logger = logging.getLogger(__name__)


# ─── Semantic Search ──────────────────────────────────────────────────────────

@router.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """Perform vector similarity search on document chunks."""
    results = await rag.semantic_search(
        query=request.query,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )

    search_results = [
        SearchResult(
            chunk=ChunkResponse(
                id=str(r["id"]),
                document_id=r["document_id"],
                content=r["content"],
                chunk_index=r["chunk_index"],
                page_number=r.get("page_number"),
                section=r.get("section"),
                score=r["score"],
            ),
            score=r["score"],
            document_filename=r.get("document_filename", "Unknown"),
        )
        for r in results
    ]

    return SearchResponse(
        query=request.query,
        results=search_results,
        total=len(search_results),
        search_type="semantic",
    )


# ─── Graph Search ─────────────────────────────────────────────────────────────

@router.post("/search/graph", response_model=SearchResponse)
async def graph_search(
    request: SearchRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """Search the knowledge graph for entities related to the query."""
    entity_ids = rag.graph_builder.find_entities_in_text(request.query)
    context_items = rag.graph_builder.get_related_context(entity_ids, depth=2)

    results = [
        SearchResult(
            chunk=ChunkResponse(
                id=f"graph_{i}",
                document_id="knowledge_graph",
                content=item["text"],
                chunk_index=i,
                page_number=None,
                section=None,
                score=1.0,
            ),
            score=1.0,
            document_filename="Knowledge Graph",
        )
        for i, item in enumerate(context_items[:request.top_k])
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
        search_type="graph",
    )


# ─── Chat (GraphRAG v2) ──────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """Ask a question using the full GraphRAG pipeline (non-streaming).
    Supports search_type: local, global, hybrid, or auto."""
    try:
        # Convert search_type
        search_type = None
        if request.search_type and request.search_type.value != "auto":
            search_type = request.search_type.value

        result = await rag.answer(
            question=request.question,
            top_k=request.top_k,
            use_graph=request.use_graph,
            document_ids=request.document_ids,
            history=request.history,
            search_type=search_type,
        )

        citations = [
            Citation(
                document_id=c["document_id"],
                document_filename=c["document_filename"],
                chunk_id=c["chunk_id"],
                chunk_index=c["chunk_index"],
                page_number=c.get("page_number"),
                relevance_score=c["relevance_score"],
                excerpt=c["excerpt"],
            )
            for c in result["citations"]
        ]

        graph_ctx = GraphContext(
            entities=result["graph_context"].get("entities", []),
            relations=result["graph_context"].get("relations", []),
        )

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            citations=citations,
            graph_context=graph_ctx,
            semantic_chunks_used=result["semantic_chunks_used"],
            graph_nodes_used=result["graph_nodes_used"],
            model_used=result["model_used"],
            retrieval_mode=result["retrieval_mode"],
            query_type=result.get("query_type", "hybrid"),
            confidence_score=result.get("confidence_score", 0.0),
            confidence_calibrated=result.get("confidence_calibrated", False),
            timings_ms=result.get("timings_ms", {}),
            warnings=result.get("warnings", []),
            groundedness_score=result.get("groundedness_score", 0.0),
            claim_support=result.get("claim_support", []),
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """Ask a question using GraphRAG with real-time SSE token streaming.
    Supports search_type: local, global, hybrid, or auto."""
    try:
        search_type = None
        if request.search_type and request.search_type.value != "auto":
            search_type = request.search_type.value

        generator = rag.answer_stream(
            question=request.question,
            top_k=request.top_k,
            use_graph=request.use_graph,
            document_ids=request.document_ids,
            history=request.history,
            search_type=search_type,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG streaming error: {str(e)}")


# ─── Monitoring ───────────────────────────────────────────────────────────────

@router.get("/monitoring/stats", response_model=MonitoringStats)
async def get_monitoring_stats(
    monitor: Monitor = Depends(get_monitor),
):
    """Get runtime monitoring statistics: latency, error rates, query distribution."""
    return monitor.get_stats()


@router.get("/monitoring/queries")
async def get_recent_queries(
    limit: int = Query(20, ge=1, le=100),
    monitor: Monitor = Depends(get_monitor),
):
    """Get recent query logs for debugging and analysis."""
    return {"queries": monitor.get_recent_queries(limit=limit)}
