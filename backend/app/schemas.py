"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class ProcessingStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class SearchType(str, Enum):
    LOCAL  = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    AUTO   = "auto"


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
    index_version: int = 0
    error_message: Optional[str]
    progress_stage: str = "queued"
    progress_current: int = 0
    progress_total: int = 0
    heartbeat_at: Optional[datetime] = None
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
    search_type: SearchType = Field(default=SearchType.AUTO, description="Search mode: local, global, hybrid, or auto")


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
    PROJECT      = "PROJECT"
    TECHNOLOGY   = "TECHNOLOGY"
    EVENT        = "EVENT"
    DOCUMENT     = "DOCUMENT"


class RelationType(str, Enum):
    WORKS_AT         = "WORKS_AT"
    BELONGS_TO       = "BELONGS_TO"
    HAS_PREREQUISITE = "HAS_PREREQUISITE"
    MENTIONS         = "MENTIONS"
    RELATED_TO       = "RELATED_TO"
    LOCATED_IN       = "LOCATED_IN"
    MANAGES          = "MANAGES"
    USES             = "USES"
    CREATED_BY       = "CREATED_BY"
    DEPENDS_ON       = "DEPENDS_ON"
    HAS_RISK         = "HAS_RISK"
    PART_OF          = "PART_OF"


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    document_ids: List[str] = Field(default_factory=list)
    properties: dict = Field(default_factory=dict)
    community_id: Optional[int] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0
    document_ids: List[str] = Field(default_factory=list)
    description: str = ""


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_nodes: int
    total_edges: int


class EntityQueryRequest(BaseModel):
    entity_name: str
    entity_type: Optional[str] = None
    depth: int = Field(default=2, ge=1, le=4)


# ─── Community Schemas ────────────────────────────────────────────────────────

class CommunityReport(BaseModel):
    community_id: int
    level: int = 0
    title: str
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    main_entities: List[str] = Field(default_factory=list)
    importance_score: float = 0.0


class CommunityListResponse(BaseModel):
    communities: List[CommunityReport]
    total: int


# ─── Chat Schemas ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: List[ChatMessage] = Field(default_factory=list)
    use_graph: bool = True
    top_k: int = Field(default=5, ge=1, le=10)
    document_ids: Optional[List[str]] = None
    search_type: SearchType = Field(default=SearchType.AUTO, description="Search mode: local, global, hybrid, or auto")


class Citation(BaseModel):
    document_id: str
    document_filename: str
    chunk_id: str
    chunk_index: int
    page_number: Optional[int]
    relevance_score: float
    excerpt: str


class GraphContext(BaseModel):
    entities: List[str] = Field(default_factory=list)
    relations: List[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    graph_context: Optional[GraphContext]
    semantic_chunks_used: int
    graph_nodes_used: int
    model_used: str
    retrieval_mode: str  # "semantic" | "graph" | "hybrid" | "global"
    query_type: str = "hybrid"  # "local" | "global" | "hybrid"
    confidence_score: float = 0.0
    confidence_calibrated: bool = False
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    groundedness_score: float = 0.0
    claim_support: List[Dict[str, Any]] = Field(default_factory=list)


# ─── Monitoring Schemas ───────────────────────────────────────────────────────

class MonitoringStats(BaseModel):
    total_queries: int
    error_count: int
    error_rate: float
    avg_latency_ms: float
    avg_confidence: float
    query_type_distribution: Dict[str, int]
    retrieval_mode_distribution: Dict[str, int]
    stage_latencies: Dict[str, Any]
    recent_queries_count: int


class QueryLogEntry(BaseModel):
    timestamp: str
    question: str
    query_type: str
    retrieval_mode: str
    timings_ms: Dict[str, float]
    semantic_chunks_used: int
    graph_nodes_used: int
    confidence_score: float
    success: bool
    error: Optional[str] = None


# Evaluation schemas
class EvaluationCase(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    answer: str = ""
    reference_answer: str = ""
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_source_ids: List[str] = Field(default_factory=list)
    expected_no_answer: bool = False
    predicted_no_answer: bool = False
    extracted_entities: List[str] = Field(default_factory=list)
    expected_entities: List[str] = Field(default_factory=list)
    extracted_relations: List[str] = Field(default_factory=list)
    expected_relations: List[str] = Field(default_factory=list)


class EvaluationRequest(BaseModel):
    cases: List[EvaluationCase] = Field(..., min_length=1, max_length=1000)


class EvaluationCaseResult(BaseModel):
    question: str
    metrics: Dict[str, float]


class EvaluationResponse(BaseModel):
    total_cases: int
    aggregate: Dict[str, float]
    cases: List[EvaluationCaseResult]
