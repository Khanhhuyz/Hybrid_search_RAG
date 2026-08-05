import unittest

from app.services.ingestion import sample_chunks


class IngestionSamplingTests(unittest.TestCase):
    def test_small_documents_are_not_sampled(self):
        chunks = list(range(4))
        self.assertIs(sample_chunks(chunks, 10), chunks)

    def test_large_documents_are_evenly_sampled(self):
        sampled = sample_chunks(list(range(1000)), 5)
        self.assertEqual(sampled, [0, 250, 500, 749, 999])

    def test_single_chunk_limit_is_supported(self):
        self.assertEqual(sample_chunks([1, 2, 3], 1), [1])
