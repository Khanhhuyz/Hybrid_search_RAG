"""
Documents API
Handles file upload, listing, retrieval, and deletion.
"""
import uuid
import logging
import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.config import settings
from app.database import get_db, DocumentModel, ChunkModel, ProcessingStatus
from app.schemas import DocumentResponse, DocumentListResponse
from app.services.document_processor import DocumentProcessor
from app.services.chunker import TextChunker
from app.services.embedder import EmbedderService
from app.services.vector_store import VectorStoreService
from app.services.graph_builder import GraphBuilderService
from app.services.rag_pipeline import RAGPipeline
from app.dependencies import get_embedder, get_vector_store, get_graph_builder, get_rag_pipeline
from app.services import ingestion

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)

processor   = DocumentProcessor()
chunker     = TextChunker()


# ─── Upload ───────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    embedder: EmbedderService = Depends(get_embedder),
    vector_store: VectorStoreService = Depends(get_vector_store),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """Upload a document and trigger async processing pipeline."""
    # Clear pipeline cache on new document upload
    rag.clear_cache()
    # ── Validation ────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read file incrementally to check size limit without memory exhaustion
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    chunks = []
    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunk

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_MB} MB",
            )
        chunks.append(chunk)

    content = b"".join(chunks)

    # ── Persist file ──────────────────────────────────────────────────────────
    doc_id   = str(uuid.uuid4())
    filename = f"{doc_id}{suffix}"
    file_path = settings.UPLOAD_DIR / filename
    file_path.write_bytes(content)

    # ── Create DB record ──────────────────────────────────────────────────────
    doc = DocumentModel(
        id=doc_id,
        filename=filename,
        original_name=file.filename,
        file_type=suffix,
        file_size=len(content),
        file_path=str(file_path),
        status=ProcessingStatus.PENDING,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # ── Queue background processing ───────────────────────────────────────────
    ingestion.schedule(doc_id, file_path, suffix, file.filename, embedder, vector_store, graph_builder)

    return doc


# ─── List ─────────────────────────────────────────────────────────────────────

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(DocumentModel).offset(skip).limit(limit).order_by(DocumentModel.created_at.desc())
    )
    docs = result.scalars().all()

    count_result = await db.execute(select(DocumentModel))
    total = len(count_result.scalars().all())

    return {"documents": docs, "total": total}


# ─── Get Single ───────────────────────────────────────────────────────────────

@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    doc = await _get_doc_or_404(db, doc_id)
    return doc


# ─── Get Status ───────────────────────────────────────────────────────────────

@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch processing status and progress metadata for a document."""
    doc = await _get_doc_or_404(db, doc_id)
    return {
        "id": doc.id,
        "status": doc.status,
        "chunk_count": doc.chunk_count,
        "entity_count": doc.entity_count,
        "index_version": doc.index_version,
        "error_message": doc.error_message,
        "progress_stage": doc.progress_stage,
        "progress_current": doc.progress_current,
        "progress_total": doc.progress_total,
        "heartbeat_at": doc.heartbeat_at,
    }


@router.post("/{doc_id}/retry", response_model=DocumentResponse, status_code=202)
async def retry_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    embedder: EmbedderService = Depends(get_embedder),
    vector_store: VectorStoreService = Depends(get_vector_store),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    doc = await _get_doc_or_404(db, doc_id)
    if doc.status == ProcessingStatus.COMPLETED and doc.index_version >= settings.INDEX_SCHEMA_VERSION:
        if not doc.error_message:
            raise HTTPException(status_code=409, detail="Completed documents do not need retry")
        # Vector indexing is already checkpointed. Reuse chunks/embeddings and retry graph only.
        doc.status = ProcessingStatus.PENDING
        doc.progress_stage = "queued"
        doc.progress_current = 0
        doc.progress_total = 0
        doc.error_message = None
        await db.commit()
    else:
        try:
            await ingestion.reset_document(db, doc, vector_store, graph_builder)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    await db.refresh(doc)
    ingestion.schedule(doc.id, doc.file_path, doc.file_type, doc.original_name, embedder, vector_store, graph_builder)
    return doc


@router.post("/{doc_id}/reindex", response_model=DocumentResponse, status_code=202)
async def reindex_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    embedder: EmbedderService = Depends(get_embedder),
    vector_store: VectorStoreService = Depends(get_vector_store),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    """Explicitly rebuild chunks, vectors, and graph provenance with the current schema."""
    doc = await _get_doc_or_404(db, doc_id)
    try:
        await ingestion.reset_document(db, doc, vector_store, graph_builder)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    rag.clear_cache()
    ingestion.schedule(
        doc.id, doc.file_path, doc.file_type, doc.original_name,
        embedder, vector_store, graph_builder,
    )
    await db.refresh(doc)
    return doc


# ─── Delete ───────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStoreService = Depends(get_vector_store),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
    rag: RAGPipeline = Depends(get_rag_pipeline),
):
    doc = await _get_doc_or_404(db, doc_id)

    # Clear RAG cache so old questions don't return deleted doc context
    rag.clear_cache()

    # Delete vectors
    try:
        await vector_store.delete_by_document(doc_id)
    except Exception as e:
        logger.warning(f"Failed to delete vectors for {doc_id}: {e}")

    # Delete graph entities
    chunk_rows = await db.execute(select(ChunkModel.id).where(ChunkModel.document_id == doc_id))
    graph_builder.delete_document_entities(doc_id, chunk_ids=list(chunk_rows.scalars().all()))

    # Delete file
    file_path = Path(doc.file_path)
    if file_path.exists():
        file_path.unlink()

    # Delete chunks
    await db.execute(delete(ChunkModel).where(ChunkModel.document_id == doc_id))

    # Delete document record
    await db.delete(doc)
    await db.commit()


# ─── Document Chunks ─────────────────────────────────────────────────────────

@router.get("/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch all stored chunks for a specific document."""
    doc = await _get_doc_or_404(db, doc_id)
    result = await db.execute(
        select(ChunkModel)
        .where(ChunkModel.document_id == doc_id)
        .order_by(ChunkModel.chunk_index.asc())
    )
    chunks = result.scalars().all()
    return {
        "document_id": doc.id,
        "filename": doc.original_name,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "content": c.content,
                "page_number": c.page_number,
                "section": c.section,
                "page_end": c.page_end,
                "parent_id": c.parent_id,
                "chunk_type": c.chunk_type,
                "metadata": json.loads(c.metadata_json) if c.metadata_json else {},
            }
            for c in chunks
        ],
    }


# ─── Raw Document File Stream ──────────────────────────────────────────────────

from fastapi.responses import FileResponse

@router.get("/{doc_id}/file")
async def get_document_file(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Serve the raw original document file (PDF, TXT, MD, DOCX)."""
    doc = await _get_doc_or_404(db, doc_id)
    file_path = Path(doc.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    media_type = media_types.get(doc.file_type.lower(), "application/octet-stream")
    return FileResponse(path=file_path, filename=doc.original_name, media_type=media_type)


# ─── Background Processing Task ───────────────────────────────────────────────

async def _process_document(
    doc_id: str,
    file_path: Path,
    file_type: str,
    original_name: str,
    embedder: EmbedderService,
    vector_store: VectorStoreService,
    graph_builder: GraphBuilderService,
):
    """Full processing pipeline: extract → chunk → embed → graph."""
    from app.database import AsyncSessionLocal, DocumentModel, ChunkModel, ProcessingStatus
    from sqlalchemy import update

    async with AsyncSessionLocal() as db:
        try:
            # ── Update status → PROCESSING ─────────────────────────────────
            await db.execute(
                update(DocumentModel)
                .where(DocumentModel.id == doc_id)
                .values(status=ProcessingStatus.PROCESSING)
            )
            await db.commit()

            # ── Extract Text & Chunk (in threadpool to avoid blocking event loop) ──
            raw_text = await asyncio.to_thread(processor.extract_text, file_path, file_type)
            clean_text = processor.normalize(raw_text)

            # ── Chunk ──────────────────────────────────────────────────────
            chunks = await asyncio.to_thread(chunker.chunk_document, clean_text, doc_id, original_name)
            if not chunks:
                raise ValueError("No text could be extracted from the document")

            # ── Store chunks in DB ─────────────────────────────────────────
            chunk_models = [
                ChunkModel(
                    id=c["id"],
                    document_id=doc_id,
                    content=c["content"],
                    chunk_index=c["chunk_index"],
                    page_number=c.get("page_number"),
                    section=c.get("section"),
                    char_start=c.get("char_start"),
                    char_end=c.get("char_end"),
                    token_count=c.get("token_count"),
                )
                for c in chunks
            ]
            db.add_all(chunk_models)
            await db.commit()

            # ── Embed ──────────────────────────────────────────────────────
            texts      = [c["content"] for c in chunks]
            embeddings = await embedder.embed_batch(texts)
            await vector_store.upsert_chunks(chunks, embeddings)

            # ── Mark chunks as embedded ────────────────────────────────────
            from sqlalchemy import update as sa_update
            await db.execute(
                sa_update(ChunkModel)
                .where(ChunkModel.document_id == doc_id)
                .values(embedded=True)
            )
            await db.commit()

            # ── Knowledge Graph ────────────────────────────────────────────
            entity_count = 0
            try:
                entity_count = await graph_builder.process_chunks(chunks)
            except Exception as graph_error:
                # Vector indexing remains useful when the optional graph backend is degraded.
                logger.warning(
                    "Graph indexing failed for %s; completing vector indexing: %s",
                    doc_id,
                    graph_error,
                )

            # ── Update status → COMPLETED ──────────────────────────────────
            await db.execute(
                update(DocumentModel)
                .where(DocumentModel.id == doc_id)
                .values(
                    status=ProcessingStatus.COMPLETED,
                    chunk_count=len(chunks),
                    entity_count=entity_count,
                )
            )
            await db.commit()
            logger.info(f"Document {doc_id} processed: {len(chunks)} chunks, {entity_count} entities")

        except Exception as e:
            logger.error(f"Document processing failed for {doc_id}: {e}", exc_info=True)
            await db.execute(
                update(DocumentModel)
                .where(DocumentModel.id == doc_id)
                .values(
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)[:500],
                )
            )
            await db.commit()


# ─── Utility ──────────────────────────────────────────────────────────────────

async def _get_doc_or_404(db: AsyncSession, doc_id: str) -> DocumentModel:
    result = await db.execute(select(DocumentModel).where(DocumentModel.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return doc
