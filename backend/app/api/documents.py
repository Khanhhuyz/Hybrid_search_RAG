"""
Documents API
Handles file upload, listing, retrieval, and deletion.
"""
import shutil
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
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
from app.dependencies import get_embedder, get_vector_store, get_graph_builder

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger(__name__)

processor   = DocumentProcessor()
chunker     = TextChunker()


# ─── Upload ───────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    embedder: EmbedderService = Depends(get_embedder),
    vector_store: VectorStoreService = Depends(get_vector_store),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    """Upload a document and trigger async processing pipeline."""
    # ── Validation ────────────────────────────────────────────────────────────
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read file into memory to check size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {settings.MAX_FILE_SIZE_MB} MB",
        )

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
    background_tasks.add_task(
        _process_document,
        doc_id=doc_id,
        file_path=file_path,
        file_type=suffix,
        original_name=file.filename,
        embedder=embedder,
        vector_store=vector_store,
        graph_builder=graph_builder,
    )

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


# ─── Delete ───────────────────────────────────────────────────────────────────

@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    vector_store: VectorStoreService = Depends(get_vector_store),
    graph_builder: GraphBuilderService = Depends(get_graph_builder),
):
    doc = await _get_doc_or_404(db, doc_id)

    # Delete vectors
    try:
        await vector_store.delete_by_document(doc_id)
    except Exception as e:
        logger.warning(f"Failed to delete vectors for {doc_id}: {e}")

    # Delete graph entities
    graph_builder.delete_document_entities(doc_id)

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
    
    media_type = "application/pdf" if doc.file_type == "pdf" else "text/plain"
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
            entity_count = await graph_builder.process_chunks(chunks)

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
