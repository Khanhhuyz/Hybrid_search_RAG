"""
Vector Store Service
Manages chunk embeddings in Qdrant vector database (Local Mode).
"""
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
        settings.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        # Use local disk-based Qdrant (synchronous client)
        self.client = QdrantClient(path=str(settings.QDRANT_PATH))
        self.collection = settings.QDRANT_COLLECTION
        self.dimension  = settings.EMBEDDING_DIMENSION

    # ─── Initialization ───────────────────────────────────────────────────────

    async def ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections()
        existing = [c.name for c in collections.collections]

        if self.collection not in existing:
            self.client.create_collection(
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
    ):
        """Store chunks with their embeddings in Qdrant."""
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
            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
            logger.info(f"Upserted {len(points)} vectors to Qdrant")

    # ─── Search ───────────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        score_threshold: float = None,
    ) -> List[Dict[str, Any]]:
        """Perform cosine similarity search with optional document filter."""
        qdrant_filter = None
        if document_ids:
            qdrant_filter = Filter(
                must=[
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

        response = self.client.query_points(**kwargs)
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
        """Remove all vectors associated with a document."""
        self.client.delete(
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

    async def collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
            "status": info.status,
        }

    async def health_check(self) -> dict:
        try:
            collections = self.client.get_collections()
            return {"status": "ok", "collections": [c.name for c in collections.collections], "mode": "local"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
