"""
GRAG FastAPI Application Entry Point
"""
import asyncio
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

log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
if settings.DEBUG:
    log_level = logging.DEBUG

logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks with fault tolerance."""
    logger.info("🚀 Starting GRAG System...")

    # Initialize database tables
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")

    # Initialize all singleton services
    try:
        init_services()
        logger.info("✅ Services initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")

    # Ensure Qdrant collection exists & sync active docs
    try:
        from app.database import AsyncSessionLocal, DocumentModel
        from sqlalchemy import select
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(DocumentModel.id))
            active_ids = set(res.scalars().all())

        vs = get_vector_store()
        if vs:
            await vs.ensure_collection()
            await vs.purge_orphaned_vectors(active_ids)
            logger.info("✅ Vector store ready & synced")
    except Exception as e:
        logger.warning(f"⚠️ Vector store degraded or not reachable on startup: {e}")

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

# ─── Author Watermark Middleware ─────────────────────────────────────────────

@app.middleware("http")
async def add_author_watermark(request, call_next):
    response = await call_next(request)
    response.headers["X-Author"] = "QuiNC (quinc-fptu) - CC BY-NC 4.0"
    return response

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

    ollama_status = "unavailable"
    qdrant_status = "unavailable"

    if embedder:
        try:
            ollama_status = await asyncio.wait_for(embedder.health_check(), timeout=3.0)
        except Exception as e:
            logger.warning(f"Ollama health check timed out or failed: {e}")
            ollama_status = f"error: {str(e)}"

    if vector_store:
        try:
            qdrant_status = await asyncio.wait_for(vector_store.health_check(), timeout=3.0)
        except Exception as e:
            logger.warning(f"Qdrant health check timed out or failed: {e}")
            qdrant_status = f"error: {str(e)}"

    is_ollama_ok = ollama_status == "ok" or (isinstance(ollama_status, dict) and ollama_status.get("status") == "ok")
    is_qdrant_ok = qdrant_status == "ok" or (isinstance(qdrant_status, dict) and qdrant_status.get("status") == "ok")
    
    overall_status = "ok" if (is_ollama_ok and is_qdrant_ok) else "degraded"

    return {
        "status": overall_status,
        "version": settings.APP_VERSION,
        "author": "QuiNC (quinc-fptu)",
        "license": "CC BY-NC 4.0",
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
        "author":  "QuiNC (quinc-fptu)",
        "license": "CC BY-NC 4.0",
        "docs":    "/docs",
        "api":     settings.API_PREFIX,
    }
