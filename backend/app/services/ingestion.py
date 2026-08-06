"""Recoverable, checkpointed document ingestion jobs."""
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import delete, select, update
from app.config import settings
from app.database import AsyncSessionLocal, ChunkModel, DocumentModel, ProcessingStatus
from app.services.chunker import TextChunker
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
processor, chunker = DocumentProcessor(), TextChunker()
_jobs: dict[str, asyncio.Task] = {}
_worker_lock = asyncio.Semaphore(1)

async def _progress(db, doc_id, stage, current=0, total=0, **extra):
    extra.update(progress_stage=stage, progress_current=current, progress_total=total,
                 heartbeat_at=datetime.now(timezone.utc))
    await db.execute(update(DocumentModel).where(DocumentModel.id == doc_id).values(**extra))
    await db.commit()

def _as_dict(c, document_filename=""):
    return {"id": c.id, "document_id": c.document_id, "content": c.content,
            "chunk_index": c.chunk_index, "page_number": c.page_number, "section": c.section,
            "page_end": c.page_end, "parent_id": c.parent_id, "chunk_type": c.chunk_type,
            "parent_content": c.parent_content,
            "metadata": json.loads(c.metadata_json) if c.metadata_json else {},
            "document_filename": document_filename,
            "char_start": c.char_start, "char_end": c.char_end, "token_count": c.token_count}

def sample_chunks(chunks, limit):
    if limit <= 0 or len(chunks) <= limit:
        return chunks
    if limit == 1:
        return chunks[:1]
    return [chunks[round(i * (len(chunks) - 1) / (limit - 1))] for i in range(limit)]

async def process_document(doc_id, file_path, file_type, original_name, embedder, vector_store, graph_builder):
    async with _worker_lock, AsyncSessionLocal() as db:
        try:
            await _progress(db, doc_id, "extracting", status=ProcessingStatus.PROCESSING, error_message=None)
            result = await db.execute(select(ChunkModel).where(ChunkModel.document_id == doc_id).order_by(ChunkModel.chunk_index))
            models = list(result.scalars().all())
            if not models:
                pages = await asyncio.to_thread(processor.extract_pages, Path(file_path), file_type)
                chunks = await asyncio.to_thread(chunker.chunk_pages, pages, doc_id, original_name)
                if not chunks:
                    raise ValueError("No text could be extracted from the document")
                fields = ("id", "document_id", "content", "chunk_index", "page_number", "page_end", "section", "parent_id", "chunk_type", "parent_content", "char_start", "char_end", "token_count")
                models = [ChunkModel(**{key: item.get(key) for key in fields}, metadata_json=json.dumps(item.get("metadata", {}), ensure_ascii=False)) for item in chunks]
                db.add_all(models)
                await db.commit()
            chunks, total = [_as_dict(c, original_name) for c in models], len(models)
            pending = [(c, m) for c, m in zip(chunks, models) if not m.embedded]
            completed = total - len(pending)
            await _progress(db, doc_id, "embedding", completed, total, chunk_count=total)
            size = settings.EMBEDDING_BATCH_SIZE
            for offset in range(0, len(pending), size):
                batch = pending[offset:offset + size]
                batch_chunks = [x[0] for x in batch]
                vectors = await embedder.embed_batch([x["content"] for x in batch_chunks])
                await vector_store.upsert_chunks(batch_chunks, vectors)
                await db.execute(update(ChunkModel).where(ChunkModel.id.in_([x[1].id for x in batch])).values(embedded=True))
                completed += len(batch)
                await _progress(db, doc_id, "embedding", completed, total, chunk_count=total)
            graph_chunks = sample_chunks(
                [c for c in chunks if c.get("chunk_type") == "text"],
                settings.GRAPH_MAX_CHUNKS_PER_DOCUMENT,
            )
            await _progress(db, doc_id, "graph", 0, len(graph_chunks), chunk_count=total)
            entities, warning = 0, None
            try:
                if graph_builder.neo4j.health_check() == "ok":
                    async def graph_progress(current, total_graph):
                        await _progress(db, doc_id, "graph", current, total_graph, chunk_count=total)
                    entities = await graph_builder.process_chunks(graph_chunks, graph_progress)
                else:
                    warning = "Vector indexing completed; Neo4j unavailable, graph indexing skipped."
            except Exception as exc:
                warning = f"Vector indexing completed; graph indexing failed: {exc}"[:500]
                logger.warning("Graph indexing failed for %s: %s", doc_id, exc)
            await _progress(db, doc_id, "completed", total, total, status=ProcessingStatus.COMPLETED,
                            chunk_count=total, entity_count=entities, error_message=warning,
                            index_version=settings.INDEX_SCHEMA_VERSION)
        except asyncio.CancelledError:
            logger.info("Ingestion %s interrupted; it will resume on startup", doc_id)
            raise
        except Exception as exc:
            logger.exception("Document processing failed for %s", doc_id)
            await _progress(db, doc_id, "failed", status=ProcessingStatus.FAILED, error_message=str(exc)[:500])

def schedule(doc_id, file_path, file_type, original_name, embedder, vector_store, graph_builder):
    current = _jobs.get(doc_id)
    if current and not current.done():
        return current
    task = asyncio.create_task(process_document(doc_id, file_path, file_type, original_name, embedder, vector_store, graph_builder), name=f"ingest-{doc_id}")
    _jobs[doc_id] = task
    task.add_done_callback(lambda _: _jobs.pop(doc_id, None))
    return task

def is_active(doc_id): return doc_id in _jobs and not _jobs[doc_id].done()

async def recover(embedder, vector_store, graph_builder):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(DocumentModel).where(DocumentModel.status.in_([ProcessingStatus.PENDING, ProcessingStatus.PROCESSING])))
        docs = list(result.scalars().all())
        for doc in docs:
            doc.status = ProcessingStatus.PENDING
            doc.progress_stage = "queued"
            doc.heartbeat_at = datetime.now(timezone.utc)
        await db.commit()
    for doc in docs:
        schedule(doc.id, doc.file_path, doc.file_type, doc.original_name, embedder, vector_store, graph_builder)
    logger.info("Recovered %d incomplete ingestion job(s)", len(docs))

async def reset_document(db, doc, vector_store, graph_builder):
    if is_active(doc.id):
        raise RuntimeError("Document is already being processed")
    chunk_rows = await db.execute(select(ChunkModel.id).where(ChunkModel.document_id == doc.id))
    chunk_ids = list(chunk_rows.scalars().all())
    await vector_store.delete_by_document(doc.id)
    try:
        graph_builder.delete_document_entities(doc.id, chunk_ids=chunk_ids)
    except Exception as exc:
        logger.warning("Graph cleanup failed for %s: %s", doc.id, exc)
    await db.execute(delete(ChunkModel).where(ChunkModel.document_id == doc.id))
    doc.status, doc.progress_stage, doc.error_message = ProcessingStatus.PENDING, "queued", None
    doc.chunk_count = doc.entity_count = doc.progress_current = doc.progress_total = 0
    doc.index_version = 0
    await db.commit()

async def shutdown():
    tasks = list(_jobs.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
