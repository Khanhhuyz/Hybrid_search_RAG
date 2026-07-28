"""
GRAG FastAPI Application Entry Point
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.dependencies import init_services, get_vector_store
from app.api import documents, search, graph

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("🚀 Starting GRAG System...")

    # Initialize database tables
    await init_db()
    logger.info("✅ Database initialized")

    # Initialize all singleton services
    init_services()
    logger.info("✅ Services initialized")

    # Ensure Qdrant collection exists
    vs = get_vector_store()
    await vs.ensure_collection()
    logger.info("✅ Vector store ready")

    logger.info(f"🌐 GRAG API running on http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}")
    yield

    logger.info("👋 Shutting down GRAG System...")


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Mini GraphRAG System — Semantic Search + Knowledge Graph + LLM",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(documents.router, prefix=settings.API_PREFIX)
app.include_router(search.router,    prefix=settings.API_PREFIX)
app.include_router(graph.router,     prefix=settings.API_PREFIX)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    from app.dependencies import get_embedder, get_vector_store
    embedder     = get_embedder()
    vector_store = get_vector_store()

    ollama_status = await embedder.health_check()
    qdrant_status = await vector_store.health_check()

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "services": {
            "ollama":  ollama_status,
            "qdrant":  qdrant_status,
        },
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name":    settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs":    "/docs",
        "api":     settings.API_PREFIX,
    }
