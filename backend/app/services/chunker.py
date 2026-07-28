"""
Text Chunker Service
Splits documents into overlapping chunks preserving metadata.
"""
import re
import uuid
import logging
from typing import List, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Recursive character-level chunker with overlap support.
    Respects sentence boundaries where possible.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size    = chunk_size    or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_filename: str,
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks and return list of chunk dicts
        with full metadata.
        """
        raw_chunks = self._split_text(text)
        chunks = []
        char_cursor = 0

        for idx, chunk_text in enumerate(raw_chunks):
            # Find where this chunk starts in the original text
            start = text.find(chunk_text[:50], char_cursor)
            if start == -1:
                start = char_cursor
            end = start + len(chunk_text)

            # Approximate page number (assume ~3000 chars per page)
            page_number = (start // 3000) + 1

            chunks.append({
                "id":            str(uuid.uuid4()),
                "document_id":   document_id,
                "content":       chunk_text.strip(),
                "chunk_index":   idx,
                "page_number":   page_number,
                "section":       self._detect_section(chunk_text),
                "char_start":    start,
                "char_end":      end,
                "token_count":   self._estimate_tokens(chunk_text),
                "document_filename": document_filename,
            })

            # Move cursor forward (accounting for overlap)
            char_cursor = max(char_cursor, end - self.chunk_overlap)

        logger.info(f"Chunked document {document_id} → {len(chunks)} chunks")
        return chunks

    # ─── Internal Methods ─────────────────────────────────────────────────────

    def _split_text(self, text: str) -> List[str]:
        """
        Recursively split using a hierarchy of separators.
        Tries to respect paragraph > sentence > word boundaries.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:]

        parts = text.split(sep) if sep else list(text)
        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single part exceeds chunk_size, recurse with finer separator
                if len(part) > self.chunk_size and remaining_seps:
                    sub_chunks = self._recursive_split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current)

        # Apply overlap — prepend tail of previous chunk to next
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            result.append(prev_tail + " " + chunks[i])
        return result

    def _detect_section(self, text: str) -> str:
        """Heuristically detect section heading from chunk content."""
        first_line = text.strip().split("\n")[0]
        if len(first_line) < 100 and re.match(r"^[A-Z\d].*[^.]$", first_line.strip()):
            return first_line.strip()[:80]
        return ""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars/token for English text)."""
        return max(1, len(text) // 4)
