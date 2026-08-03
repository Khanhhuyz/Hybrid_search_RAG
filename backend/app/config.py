"""
GRAG System Configuration
Central configuration management with environment variable support.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


BASE_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    # ─── Application ─────────────────────────────────────────────────
    APP_NAME: str = "GRAG - GraphRAG System"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_PREFIX: str = "/api/v1"

    # ─── Server ──────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # ─── File Storage ─────────────────────────────────────────────────
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]

    # ─── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/grag.db"

    # ─── Ollama ───────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "llama3.2"
    EMBEDDING_DIMENSION: int = 768

    # ─── Qdrant ───────────────────────────────────────────────────────
    QDRANT_PATH: Path = BASE_DIR / "data" / "qdrant_storage"
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
    GRAPH_FILE: Path = BASE_DIR / "data" / "knowledge_graph.json"

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure required directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
