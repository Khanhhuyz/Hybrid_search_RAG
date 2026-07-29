"""
Vector Store Service
Manages chunk embeddings in Qdrant vector database (Local Mode).
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Qdrant-backed vector store for chunk embeddings."""

    def __init__(self):
        if settings.QDRANT_URL and settings.QDRANT_API_KEY:
            logger.info(f"Connecting to Qdrant Cloud: {settings.QDRANT_URL}")
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            settings.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.QDRANT_PATH))
        self.collection = settings.QDRANT_COLLECTION
        self.dimension  = settings.EMBEDDING_DIMENSION

    # ─── Initialization ───────────────────────────────────────────────────────

    async def ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist (non-blocking)."""
        collections = await asyncio.to_thread(self.client.get_collections)
        existing = [c.name for c in collections.collections]

        if self.collection not in existing:
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")
        else:
            logger.info(f"Qdrant collection '{self.collection}' already exists")

    # ─── Upsert ───────────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ):
        """Store chunks with their embeddings in Qdrant in batches (non-blocking)."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = []
        for chunk, vector in zip(chunks, embeddings):
            if all(v == 0.0 for v in vector):
                logger.warning(f"Skipping zero-vector chunk: {chunk['id']}")
                continue

            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=vector,
                    payload={
                        "document_id":       chunk["document_id"],
                        "document_filename": chunk.get("document_filename", ""),
                        "content":           chunk["content"],
                        "chunk_index":       chunk["chunk_index"],
                        "page_number":       chunk.get("page_number"),
                        "section":           chunk.get("section", ""),
                    },
                )
            )

        if points:
            # Batch upsert to prevent huge memory / payload spikes
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.collection,
                    points=batch,
                )
            logger.info(f"Upserted {len(points)} vectors to Qdrant in batches of {batch_size}")

    # ─── Search ───────────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Perform cosine similarity search with optional document filter (non-blocking)."""
        qdrant_filter = None
        if document_ids:
            if len(document_ids) == 1:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_ids[0]),
                        )
                    ]
                )
            else:
                qdrant_filter = Filter(
                    should=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=did),
                        )
                        for did in document_ids
                    ]
                )

        threshold = score_threshold if score_threshold is not None else settings.SIMILARITY_THRESHOLD
        kwargs: Dict[str, Any] = {
            "collection_name": self.collection,
            "query": query_vector,
            "limit": top_k,
            "query_filter": qdrant_filter,
            "with_payload": True,
        }
        if threshold is not None and threshold > 0:
            kwargs["score_threshold"] = threshold

        response = await asyncio.to_thread(self.client.query_points, **kwargs)
        results = response.points

        return [
            {
                "id":                r.id,
                "score":             r.score,
                "document_id":       r.payload.get("document_id"),
                "document_filename": r.payload.get("document_filename"),
                "content":           r.payload.get("content"),
                "chunk_index":       r.payload.get("chunk_index"),
                "page_number":       r.payload.get("page_number"),
                "section":           r.payload.get("section"),
            }
            for r in results
        ]

    # ─── Delete ───────────────────────────────────────────────────────────────

    async def delete_by_document(self, document_id: str):
        """Remove all vectors associated with a document (non-blocking)."""
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted vectors for document: {document_id}")

    # ─── Stats ────────────────────────────────────────────────────────────────

    async def purge_orphaned_vectors(self, active_document_ids: set):
        """Remove points in Qdrant whose document_id is no longer in active_document_ids."""
        try:
            res = await asyncio.to_thread(
                self.client.scroll,
                collection_name=self.collection,
                limit=1000,
                with_payload=True,
            )
            points = res[0]
            orphaned = [
                p.id for p in points
                if p.payload and p.payload.get("document_id") not in active_document_ids
            ]
            if orphaned:
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=self.collection,
                    points_selector=orphaned,
                )
                logger.info(f"Purged {len(orphaned)} orphaned vectors from Qdrant")
        except Exception as e:
            logger.warning(f"Failed to purge orphaned vectors: {e}")

    async def collection_info(self) -> dict:
        info = await asyncio.to_thread(self.client.get_collection, self.collection)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
            "status": info.status,
        }

    async def health_check(self) -> dict:
        try:
            collections = await asyncio.to_thread(self.client.get_collections)
            return {"status": "ok", "collections": [c.name for c in collections.collections], "mode": "local"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

