"""
Entity Normalizer
Normalizes extracted entities: alias resolution, deduplication via fuzzy matching,
case normalization, and type validation.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from thefuzz import fuzz
from app.config import settings

logger = logging.getLogger(__name__)

# Common aliases — expand as needed
ALIAS_MAP = {
    "JS": "JAVASCRIPT",
    "TS": "TYPESCRIPT",
    "PY": "PYTHON",
    "AI": "ARTIFICIAL INTELLIGENCE",
    "ML": "MACHINE LEARNING",
    "DL": "DEEP LEARNING",
    "NLP": "NATURAL LANGUAGE PROCESSING",
    "CV": "COMPUTER VISION",
    "DB": "DATABASE",
    "API": "APPLICATION PROGRAMMING INTERFACE",
    "UI": "USER INTERFACE",
    "UX": "USER EXPERIENCE",
    "AWS": "AMAZON WEB SERVICES",
    "GCP": "GOOGLE CLOUD PLATFORM",
    "LLM": "LARGE LANGUAGE MODEL",
    "RAG": "RETRIEVAL AUGMENTED GENERATION",
    "VN": "VIETNAM",
    "HCM": "HO CHI MINH CITY",
    "HCMC": "HO CHI MINH CITY",
    "TP.HCM": "HO CHI MINH CITY",
    "HN": "HANOI",
}

VALID_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "COURSE", "DEPARTMENT",
    "PRODUCT", "LOCATION", "CONCEPT", "PROJECT",
    "TECHNOLOGY", "EVENT", "DOCUMENT",
}

VALID_RELATION_TYPES = {
    "WORKS_AT", "BELONGS_TO", "HAS_PREREQUISITE", "MENTIONS",
    "RELATED_TO", "LOCATED_IN", "MANAGES", "USES", "CREATED_BY",
    "DEPENDS_ON", "HAS_RISK", "PART_OF",
}

GENERIC_ENTITY_LABELS = VALID_ENTITY_TYPES | {
    "MODEL", "DATA", "DATASET", "SYSTEM", "INPUT", "OUTPUT", "RESPONSE",
    "TABLE", "CHAPTER", "BOOK", "DOCUMENT", "METHOD", "ALGORITHM",
    "USER", "CODE", "EXAMPLE", "VALUE", "RESULT", "SAMPLE",
}


class EntityNormalizer:
    """Normalize entities for consistent knowledge graph construction."""

    def __init__(
        self,
        fuzzy_threshold: int = 92,
        alias_map: Optional[Dict[str, str]] = None,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.alias_map = alias_map or ALIAS_MAP
        # Reverse alias map for lookup
        self._reverse_aliases: Dict[str, str] = {}
        for short, full in self.alias_map.items():
            self._reverse_aliases[short.upper()] = full.upper()

        # Cache of known labels for dedup matching
        self._known_labels: Dict[str, str] = {}  # normalized_label -> canonical_label

    def normalize_entities(
        self,
        entities: List[Dict],
        existing_labels: Optional[List[str]] = None,
        source_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Normalize a list of extracted entities.

        Args:
            entities: List of {"label": ..., "type": ..., ...}
            existing_labels: Labels already in the graph for dedup matching

        Returns:
            Normalized entities with resolved aliases and validated types.
        """
        if existing_labels:
            for label in existing_labels:
                self._known_labels[label.upper().strip()] = label.upper().strip()

        normalized = []
        seen_labels = set()

        for entity in entities:
            label = str(entity.get("label", "")).strip()
            etype = str(entity.get("type", "CONCEPT")).upper().strip()

            if not label:
                continue

            # 1. Case normalization
            label = label.upper().strip()

            # 2. Alias resolution
            label = self._resolve_alias(label)

            if not self._valid_label(label):
                continue
            if source_text is not None:
                original = str(entity.get("label", "")).strip()
                normalized_source = self._normalize_text(source_text)
                if self._normalize_text(original) not in normalized_source:
                    continue

            # 3. Type validation
            etype = self._validate_type(etype)

            # 4. Deduplication — fuzzy match against known labels
            canonical = self._find_duplicate(label)
            if canonical:
                label = canonical

            # Skip if already seen in this batch
            if label in seen_labels:
                continue
            seen_labels.add(label)

            # Register for future dedup
            self._known_labels[label] = label

            normalized.append({
                **entity,
                "label": label,
                "type": etype,
                "original_label": entity.get("label", ""),
            })

        return normalized

    def normalize_relationships(
        self,
        relationships: List[Dict],
        label_map: Dict[str, str],
        source_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Normalize relationship source/target to match normalized entity labels.

        Args:
            relationships: List of {"source": ..., "target": ..., "relation": ...}
            label_map: Mapping from original_label → normalized_label
        """
        normalized = []
        for rel in relationships:
            source = str(rel.get("source", "")).upper().strip()
            target = str(rel.get("target", "")).upper().strip()
            relation = str(rel.get("relation", "")).upper().strip()

            # Resolve via label_map
            source = label_map.get(source, self._resolve_alias(source))
            target = label_map.get(target, self._resolve_alias(target))

            if not source or not target or source == target:
                continue

            # Validate relation type
            relation = re.sub(r"[^A-Z_]", "", relation)
            if relation not in VALID_RELATION_TYPES:
                continue

            try:
                confidence = float(rel.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < settings.GRAPH_MIN_RELATION_CONFIDENCE:
                continue

            evidence = str(rel.get("evidence", "")).strip()
            if source_text is not None and settings.GRAPH_REQUIRE_VERBATIM_EVIDENCE:
                if not evidence or self._normalize_text(evidence) not in self._normalize_text(source_text):
                    continue

            normalized.append({
                **rel,
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": confidence,
                "evidence": evidence,
            })

        return normalized

    # ─── Internal Methods ────────────────────────────────────────────────────

    def _resolve_alias(self, label: str) -> str:
        """Resolve known aliases to canonical form."""
        return self._reverse_aliases.get(label, label)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).casefold().strip()

    @staticmethod
    def _valid_label(label: str) -> bool:
        if not (settings.GRAPH_MIN_ENTITY_LENGTH <= len(label) <= settings.GRAPH_MAX_ENTITY_LENGTH):
            return False
        if label in GENERIC_ENTITY_LABELS:
            return False
        if re.fullmatch(r"[\d\W_]+", label, flags=re.UNICODE):
            return False
        if re.search(r"https?://|www\.|[={}<>]|\b(?:none|null|true|false)\b", label, re.I):
            return False
        if len(label.split()) > 12:
            return False
        return True

    def _validate_type(self, entity_type: str) -> str:
        """Validate entity type against schema, fallback to CONCEPT."""
        if entity_type in VALID_ENTITY_TYPES:
            return entity_type
        # Try fuzzy match
        for valid_type in VALID_ENTITY_TYPES:
            if fuzz.ratio(entity_type, valid_type) > 80:
                return valid_type
        return "CONCEPT"

    def _find_duplicate(self, label: str) -> Optional[str]:
        """
        Find if label is a fuzzy duplicate of a known label.
        Returns the canonical label if found, None otherwise.
        """
        if label in self._known_labels:
            return self._known_labels[label]

        for known_label in self._known_labels:
            score = fuzz.ratio(label, known_label)
            if score >= self.fuzzy_threshold:
                logger.debug(f"Entity dedup: '{label}' → '{known_label}' (score={score})")
                return self._known_labels[known_label]

        return None

    def get_label_map(self, entities: List[Dict]) -> Dict[str, str]:
        """Build mapping from original labels to normalized labels."""
        return {
            entity.get("original_label", "").upper(): entity["label"]
            for entity in entities
            if "original_label" in entity
        }

    def clear_cache(self):
        self._known_labels.clear()
