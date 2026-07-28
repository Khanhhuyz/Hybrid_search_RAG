"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


# ─── Document Schemas ─────────────────────────────────────────────────────────

class DocumentBase(BaseModel):
    filename: str
    original_name: str
    file_type: str
    file_size: int


class DocumentResponse(DocumentBase):
    id: str
    status: ProcessingStatus
    chunk_count: int
    entity_count: int
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# ─── Chunk Schemas ────────────────────────────────────────────────────────────

class ChunkResponse(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int
    page_number: Optional[int]
    section: Optional[str]
    score: Optional[float] = None


# ─── Search Schemas ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: Optional[List[str]] = None


class SearchResult(BaseModel):
    chunk: ChunkResponse
    score: float
    document_filename: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    search_type: str


# ─── Graph Schemas ────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    PERSON       = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    COURSE       = "COURSE"
    DEPARTMENT   = "DEPARTMENT"
    PRODUCT      = "PRODUCT"
    LOCATION     = "LOCATION"
    CONCEPT      = "CONCEPT"


class RelationType(str, Enum):
    WORKS_AT      = "WORKS_AT"
    BELONGS_TO    = "BELONGS_TO"
    HAS_PREREQUISITE = "HAS_PREREQUISITE"
    MENTIONS      = "MENTIONS"
    RELATED_TO    = "RELATED_TO"
    LOCATED_IN    = "LOCATED_IN"


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    document_ids: List[str] = []
    properties: dict = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0
    document_ids: List[str] = []


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_nodes: int
    total_edges: int


class EntityQueryRequest(BaseModel):
    entity_name: str
    entity_type: Optional[str] = None
    depth: int = Field(default=2, ge=1, le=4)


# ─── Chat Schemas ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: Optional[List[ChatMessage]] = []
    use_graph: bool = True
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: Optional[List[str]] = None


class Citation(BaseModel):
    document_id: str
    document_filename: str
    chunk_id: str
    chunk_index: int
    page_number: Optional[int]
    relevance_score: float
    excerpt: str


class GraphContext(BaseModel):
    entities: List[str] = []
    relations: List[dict] = []


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    graph_context: Optional[GraphContext]
    semantic_chunks_used: int
    graph_nodes_used: int
    model_used: str
    retrieval_mode: str  # "semantic" | "graph" | "hybrid"
