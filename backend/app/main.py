"""
GRAG FastAPI Application Entry Point (v2)
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.dependencies import init_services, get_vector_store, get_graph_builder, get_embedder
from app.services import ingestion
from app.api import documents, search, graph, evaluation, agent

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
    logger.info("🚀 Starting GRAG System v2...")

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

    # Check Neo4j connectivity
    try:
        gb = get_graph_builder()
        if gb and gb.neo4j.is_connected:
            logger.info("✅ Neo4j connected")
        else:
            logger.warning("⚠️ Neo4j not connected — graph features degraded")
    except Exception as e:
        logger.warning(f"⚠️ Neo4j check failed: {e}")

    logger.info(f"🌐 GRAG API v2 running on http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}")
    try:
        await ingestion.recover(get_embedder(), get_vector_store(), get_graph_builder())
    except Exception as e:
        logger.error("Failed to recover ingestion jobs: %s", e)

    yield

    await ingestion.shutdown()
    # Shutdown: close Neo4j driver
    try:
        gb = get_graph_builder()
        if gb and gb.neo4j:
            gb.neo4j.close()
            logger.info("👋 Neo4j connection closed")
    except Exception:
        pass

    logger.info("👋 Shutting down GRAG System...")


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-Grade GraphRAG System — Semantic Search + Knowledge Graph + Communities + LLM",
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
    response.headers["X-Author"] = "HuyNNK - CC BY-NC 4.0"
    return response

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(documents.router, prefix=settings.API_PREFIX)
app.include_router(search.router,    prefix=settings.API_PREFIX)
app.include_router(graph.router,     prefix=settings.API_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_PREFIX)
app.include_router(agent.router, prefix=settings.API_PREFIX)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    from app.dependencies import get_embedder, get_vector_store, get_graph_builder

    embedder     = get_embedder()
    vector_store = get_vector_store()
    graph_builder = get_graph_builder()

    ollama_status = "unavailable"
    qdrant_status = "unavailable"
    neo4j_status  = "unavailable"

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

    if graph_builder:
        try:
            neo4j_status = graph_builder.neo4j.health_check()
        except Exception as e:
            neo4j_status = f"error: {str(e)}"

    is_ollama_ok = ollama_status == "ok" or (isinstance(ollama_status, dict) and ollama_status.get("status") == "ok")
    is_qdrant_ok = qdrant_status == "ok" or (isinstance(qdrant_status, dict) and qdrant_status.get("status") == "ok")
    is_neo4j_ok  = neo4j_status == "ok"

    if is_ollama_ok and is_qdrant_ok and is_neo4j_ok:
        overall_status = "ok"
    elif is_ollama_ok and is_qdrant_ok:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "version": settings.APP_VERSION,
        "author": "HuyNNK",
        "license": "CC BY-NC 4.0",
        "services": {
            "ollama":  ollama_status,
            "qdrant":  qdrant_status,
            "neo4j":   neo4j_status,
        },
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "name":    settings.APP_NAME,
        "version": settings.APP_VERSION,
        "author":  "HuyNNK",
        "license": "CC BY-NC 4.0",
        "docs":    "/docs",
        "api":     settings.API_PREFIX,
        "features": [
            "Document Ingestion (PDF, DOCX, TXT, MD)",
            "Intelligent Chunking with Section Detection",
            "Vector Search (Qdrant + nomic-embed-text)",
            "Knowledge Graph (Neo4j)",
            "Leiden Community Detection",
            "LLM-powered Community Reports",
            "Local Search (entity-focused)",
            "Global Search (Map-Reduce)",
            "Hybrid Search (Vector + Graph + RRF)",
            "SSE Streaming",
            "Confidence Scoring",
            "Runtime Monitoring",
        ],
    }
