"""Tests for deterministic structured graph extraction requests."""
import unittest
from unittest.mock import MagicMock

from app.services.graph_builder import GraphBuilderService, build_extraction_payload
from app.services.entity_normalizer import EntityNormalizer


class TestGraphExtraction(unittest.TestCase):
    def test_payload_forces_json_and_zero_temperature(self):
        payload = build_extraction_payload("extract")
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertFalse(payload["stream"])

    def test_parser_accepts_plain_json(self):
        service = object.__new__(GraphBuilderService)
        result = service._parse_json_response('{"entities": [], "relationships": []}')
        self.assertEqual(result, {"entities": [], "relationships": []})

    def test_parser_rejects_incomplete_shape(self):
        service = object.__new__(GraphBuilderService)
        self.assertIsNone(service._parse_json_response('{"entities": []}'))

    def test_merge_resolves_relationship_entity_ids(self):
        service = object.__new__(GraphBuilderService)
        service.normalizer = EntityNormalizer()
        service.neo4j = MagicMock()
        service.neo4j.upsert_entity.return_value = True
        extraction = {
            "entities": [
                {"id": "ALICE", "label": "Alice", "type": "PERSON"},
                {"id": "PROJECT_ALPHA", "label": "Project Alpha", "type": "PROJECT"},
            ],
            "relationships": [
                {
                    "source": "ALICE", "target": "PROJECT_ALPHA", "relation": "MANAGES",
                    "evidence": "Alice manages Project Alpha", "confidence": 0.9,
                }
            ],
        }

        added = service._merge_into_graph(
            extraction, "doc-1", "chunk-1", "Alice manages Project Alpha."
        )

        self.assertEqual(added, 2)
        service.neo4j.upsert_relationship.assert_called_once_with(
            source_id="ENTITY::ALICE",
            target_id="ENTITY::PROJECT ALPHA",
            relation="MANAGES",
            document_id="doc-1",
            description="",
            chunk_id="chunk-1",
            evidence="Alice manages Project Alpha",
            confidence=0.9,
            valid_from=None,
            valid_to=None,
        )

    def test_normalizer_rejects_generic_entities_and_out_of_schema_relations(self):
        normalizer = EntityNormalizer()
        entities = normalizer.normalize_entities(
            [
                {"label": "MODEL", "type": "PRODUCT"},
                {"label": "Python", "type": "TECHNOLOGY"},
            ],
            source_text="Python is used by the service.",
        )
        self.assertEqual([item["label"] for item in entities], ["PYTHON"])
        relations = normalizer.normalize_relationships(
            [{
                "source": "PYTHON", "target": "SERVICE", "relation": "COLLABORATES_WITH",
                "evidence": "Python is used by the service", "confidence": 0.99,
            }],
            {"PYTHON": "PYTHON", "SERVICE": "SERVICE"},
            source_text="Python is used by the service.",
        )
        self.assertEqual(relations, [])


if __name__ == "__main__":
    unittest.main()
