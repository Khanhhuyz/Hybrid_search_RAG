"""
GRAG System Configuration
Central configuration management with environment variable support.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from pathlib import Path
from typing import Optional
import os


BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    # ─── Application ─────────────────────────────────────────────────
    APP_NAME: str = "GRAG - GraphRAG System"
    APP_VERSION: str = "2.0.0"
    INDEX_SCHEMA_VERSION: int = 2
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"

    # ─── Server ──────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # ─── File Storage ─────────────────────────────────────────────────
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]

    # ─── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite:///{DATA_DIR}/grag.db"

    # ─── Ollama ───────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_DIMENSION: int = 768

    # ─── Qdrant ───────────────────────────────────────────────────────
    QDRANT_PATH: Path = DATA_DIR / "qdrant_storage"
    QDRANT_COLLECTION: str = "grag_chunks"
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: int = 6333

    # ─── Neo4j Knowledge Graph ────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "grag_password_2024"
    NEO4J_DATABASE: str = "neo4j"

    # ─── Legacy NetworkX (fallback) ───────────────────────────────────
    GRAPH_FILE: Path = DATA_DIR / "knowledge_graph.json"

    # ─── Chunking ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    # ─── Retrieval ────────────────────────────────────────────────────
    TOP_K_SEMANTIC: int = 5
    TOP_K_GRAPH: int = 5
    SIMILARITY_THRESHOLD: float = 0.0

    # ─── RAG ──────────────────────────────────────────────────────────
    MAX_CONTEXT_CHUNKS: int = 8
    MAX_CONTEXT_TOKENS: int = 4000
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    EMBEDDING_BATCH_SIZE: int = 50
    GRAPH_MAX_CHUNKS_PER_DOCUMENT: int = 120

    # Advanced retrieval. All enhancements have deterministic local fallbacks so
    # the project remains usable without downloading an additional model.
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 6
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    RRF_K: int = 60
    DENSE_WEIGHT: float = 1.0
    SPARSE_WEIGHT: float = 1.0
    GRAPH_WEIGHT: float = 0.8
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: Optional[str] = None
    RERANKER_OLLAMA_FALLBACK: bool = True
    RERANKER_OLLAMA_BATCH_SIZE: int = 8
    RERANK_MIN_SCORE: float = 0.12
    RETRIEVAL_MIN_EVIDENCE_SCORE: float = 0.18
    NO_ANSWER_CONFIDENCE_THRESHOLD: float = 0.35
    PARENT_CHUNK_SIZE: int = 3000
    GLOBAL_CONTEXT_CHUNKS: int = 16
    GLOBAL_STRUCTURE_CANDIDATES: int = 80
    STRUCTURE_WEIGHT: float = 0.7
    OCR_ENABLED: bool = True
    OCR_MIN_PAGE_CHARS: int = 40

    # Agent workspace
    AGENT_ENABLED: bool = True
    AGENT_MAX_STEPS: int = 8
    AGENT_MAX_TOOL_CALLS: int = 15
    AGENT_TIMEOUT: float = 180.0
    OUTPUT_DIR: Path = DATA_DIR / "outputs"
    MAX_OUTPUT_FILE_SIZE_MB: int = 50
    ACTIONS_ENABLED: bool = False
    ACTION_REQUIRE_APPROVAL: bool = True

    # Verification/calibration. CALIBRATION_* can be learned from a labelled
    # dataset; these defaults map evidence strength onto a conservative score.
    GROUNDEDNESS_THRESHOLD: float = 0.82
    GROUNDING_ENFORCED: bool = True
    CALIBRATION_FILE: Path = DATA_DIR / "evaluation" / "confidence_calibration.json"
    CALIBRATION_SLOPE: float = 7.0  # legacy fallback for old calibration files
    CALIBRATION_MIDPOINT: float = 0.48

    # Graph quality gates. Existing graph data must be re-indexed once after
    # changing these values because rejected nodes are not retroactively removed.
    GRAPH_MIN_ENTITY_LENGTH: int = 3
    GRAPH_MAX_ENTITY_LENGTH: int = 100
    GRAPH_MIN_RELATION_CONFIDENCE: float = 0.55
    GRAPH_REQUIRE_VERBATIM_EVIDENCE: bool = True

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str) and value.casefold() in {"release", "production", "prod"}:
            return False
        return value

    # ─── Community Detection ──────────────────────────────────────────
    COMMUNITY_RESOLUTION: float = 1.0
    COMMUNITY_MIN_SIZE: int = 2
    COMMUNITY_MAX_REPORT_TOKENS: int = 2000

    # ─── Query Processing ─────────────────────────────────────────────
    QUERY_CLASSIFICATION_ENABLED: bool = True
    GLOBAL_SEARCH_MAP_BATCH_SIZE: int = 5
    GLOBAL_SEARCH_MAX_COMMUNITIES: int = 50

    # ─── Monitoring ───────────────────────────────────────────────────
    MONITORING_ENABLED: bool = True
    MAX_QUERY_LOG_SIZE: int = 1000

settings = Settings()

# Ensure required directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
