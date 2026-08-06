"""Safety tests for vector-store reconciliation."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from qdrant_client.http.models import Distance, VectorParams

from app.services.vector_store import VectorStoreService


class TestVectorStoreSafety(unittest.IsolatedAsyncioTestCase):
    async def test_empty_active_documents_never_purges_vectors(self):
        service = object.__new__(VectorStoreService)
        service.client = MagicMock()
        service.collection = "test"

        deleted = await service.purge_orphaned_vectors(set())

        self.assertEqual(deleted, 0)
        service.client.scroll.assert_not_called()
        service.client.delete.assert_not_called()

    async def test_upsert_ensures_collection_before_writing(self):
        service = object.__new__(VectorStoreService)
        service.client = MagicMock()
        service.collection = "test"
        service.ensure_collection = AsyncMock()

        chunk = {
            "id": "chunk-1",
            "document_id": "doc-1",
            "content": "hello",
            "chunk_index": 0,
        }
        await service.upsert_chunks([chunk], [[1.0, 0.0]])

        service.ensure_collection.assert_awaited_once()
        service.client.upsert.assert_called_once()

    async def test_ensure_collection_creates_missing_collection(self):
        service = object.__new__(VectorStoreService)
        service.client = MagicMock()
        service.client.get_collections.return_value = SimpleNamespace(collections=[])
        service.collection = "test"
        service.dimension = 2
        service._collection_lock = __import__("asyncio").Lock()

        await service.ensure_collection()

        service.client.create_collection.assert_called_once()

    async def test_ensure_collection_recreates_empty_invalid_collection(self):
        service = object.__new__(VectorStoreService)
        service.client = MagicMock()
        service.client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="test")]
        )
        service.client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors={})),
            points_count=0,
        )
        service.collection = "test"
        service.dimension = 2
        service._collection_lock = __import__("asyncio").Lock()

        await service.ensure_collection()

        service.client.delete_collection.assert_called_once_with(
            collection_name="test"
        )
        service.client.create_collection.assert_called_once()

    async def test_ensure_collection_keeps_valid_collection(self):
        service = object.__new__(VectorStoreService)
        service.client = MagicMock()
        service.client.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name="test")]
        )
        service.client.get_collection.return_value = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=VectorParams(size=2, distance=Distance.COSINE)
                )
            ),
            points_count=1,
        )
        service.collection = "test"
        service.dimension = 2
        service._collection_lock = __import__("asyncio").Lock()

        await service.ensure_collection()

        service.client.delete_collection.assert_not_called()
        service.client.create_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
