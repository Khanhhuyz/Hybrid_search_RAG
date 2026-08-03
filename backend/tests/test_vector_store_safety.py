"""Safety tests for vector-store reconciliation."""
import unittest
from unittest.mock import MagicMock

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


if __name__ == "__main__":
    unittest.main()
