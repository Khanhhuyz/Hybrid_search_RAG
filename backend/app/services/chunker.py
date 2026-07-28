"""
Text Chunker Service
Intelligent Semantic & Structural Chunker that splits documents along
heading boundaries, paragraph units, and preserves section context in sub-chunks.
"""
import re
import uuid
import logging
from typing import List, Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Intelligent Structural & Semantic Chunker.
    - Preserves Markdown headers & section structure.
    - Prepends section titles to sub-chunks to maintain semantic context.
    - Respects paragraph, sentence, and list boundaries.
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
        Split text into intelligent semantic chunks with preserved section headings.
        """
        sections = self._split_by_sections(text)
        raw_chunks = []

        for section_title, section_text in sections:
            sub_chunks = self._split_section_content(section_text, section_title)
            for sc in sub_chunks:
                if sc.strip():
                    raw_chunks.append((section_title, sc.strip()))

        chunks = []
        char_cursor = 0

        for idx, (section_title, chunk_text) in enumerate(raw_chunks):
            # Find approximate char position in original text
            start = text.find(chunk_text[:40], char_cursor)
            if start == -1:
                start = char_cursor
            end = start + len(chunk_text)

            # Approximate page number (assume ~3000 chars per page)
            page_number = (start // 3000) + 1

            chunks.append({
                "id":                str(uuid.uuid4()),
                "document_id":       document_id,
                "content":           chunk_text,
                "chunk_index":       idx,
                "page_number":       page_number,
                "section":           section_title or self._detect_section(chunk_text),
                "char_start":        start,
                "char_end":          end,
                "token_count":       self._estimate_tokens(chunk_text),
                "document_filename": document_filename,
                "origin_sig":        "quinc-fptu-cc-by-nc-4.0",
            })

            char_cursor = max(char_cursor, end - self.chunk_overlap)

        logger.info(f"Intelligent Chunked document {document_id} → {len(chunks)} chunks")
        return chunks

    # ─── Internal Structural & Semantic Chunking Logic ───────────────────────

    def _split_by_sections(self, text: str) -> List[tuple]:
        """
        Split text by Markdown headings (#, ##, ###) or major structural breaks.
        Returns a list of (section_title, section_content) tuples.
        """
        lines = text.split("\n")
        sections = []
        current_title = ""
        current_lines = []

        heading_pattern = re.compile(r"^(#{1,6}\s+.*|[A-Z0-9\.\s\-]{3,80}:)$")

        for line in lines:
            line_str = line.strip()
            if heading_pattern.match(line_str):
                # Save previous section
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                    current_lines = []
                current_title = line_str.lstrip("#").strip()
            current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        return sections if sections else [("", text)]

    def _split_section_content(self, text: str, section_title: str) -> List[str]:
        """
        Recursively split section text into semantic sub-chunks.
        Prepends section title context if available.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        raw_parts = self._recursive_split(text, separators)

        # Prepend section heading context if chunk is a sub-chunk of a section
        final_chunks = []
        header_prefix = f"[{section_title}]\n" if section_title and len(section_title) < 100 else ""

        for part in raw_parts:
            if header_prefix and not part.startswith(header_prefix):
                final_chunks.append(f"{header_prefix}{part}")
            else:
                final_chunks.append(part)

        return final_chunks

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
                if len(part) > self.chunk_size and remaining_seps:
                    sub_chunks = self._recursive_split(part, remaining_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current)

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
        first_line = text.strip().split("\n")[0]
        if len(first_line) < 100 and re.match(r"^[A-Z\d].*[^.]$", first_line.strip()):
            return first_line.strip()[:80]
        return ""

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

