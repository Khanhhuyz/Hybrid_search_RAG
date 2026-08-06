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
        sections, _ = self._split_structured_sections(text)
        raw_chunks = []

        for section in sections:
            section_title = section["title"]
            section_text = section["content"]
            heading_path = section["heading_path"]
            is_toc = self.is_toc_text(section_text)
            sub_chunks = self._split_section_content(
                section_text, " > ".join(heading_path) or section_title
            )
            for sc in sub_chunks:
                if sc.strip():
                    raw_chunks.append((section, sc.strip(), is_toc))

        chunks = []
        char_cursor = 0

        for idx, (section, chunk_text, is_toc) in enumerate(raw_chunks):
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
                "section":           section["title"],
                "page_end":          page_number,
                "parent_id":         f"{document_id}:section:{section['ordinal']}",
                "parent_content":    section["content"][: settings.PARENT_CHUNK_SIZE],
                "chunk_type":        "toc" if is_toc else "text",
                "char_start":        start,
                "char_end":          end,
                "token_count":       self._estimate_tokens(chunk_text),
                "document_filename": document_filename,
                "metadata": {
                    "heading_path": section["heading_path"],
                    "heading_level": section["level"],
                    "chapter": section["heading_path"][0] if section["heading_path"] else None,
                    "document_filename": document_filename,
                    "is_toc": is_toc,
                    "structure_version": 2,
                },
                "origin_sig":        "quinc-fptu-cc-by-nc-4.0",
            })

            char_cursor = max(char_cursor, end - self.chunk_overlap)

        logger.info(f"Intelligent Chunked document {document_id} → {len(chunks)} chunks")
        return chunks

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        document_id: str,
        document_filename: str,
    ) -> List[Dict[str, Any]]:
        """Create searchable child chunks with exact pages and retrievable parents."""
        output: List[Dict[str, Any]] = []
        global_index = 0
        char_offset = 0
        heading_path: List[str] = []
        for page in pages:
            page_number = int(page.get("page_number", 1))
            text = page.get("text", "") or ""
            page_is_toc = self.is_toc_text(text)
            sections, next_heading_path = self._split_structured_sections(text, heading_path)
            if not page_is_toc:
                heading_path = next_heading_path
            for section in sections:
                section_title = section["title"]
                section_text = section["content"]
                if not section_text.strip():
                    continue
                # A parent is a coherent page/section window, while children stay
                # small enough for accurate retrieval and reranking.
                parent_parts = self._recursive_split(section_text, ["\n\n", "\n", ". ", " ", ""])
                grouped, current = [], ""
                for part in parent_parts:
                    if current and len(current) + len(part) + 2 > settings.PARENT_CHUNK_SIZE:
                        grouped.append(current)
                        current = ""
                    current = f"{current}\n\n{part}".strip()
                if current:
                    grouped.append(current)
                for parent_text in grouped:
                    parent_id = str(uuid.uuid4())
                    displayed_path = " > ".join(section["heading_path"])
                    children = self._split_section_content(
                        parent_text, displayed_path or section_title
                    )
                    for child in children:
                        child = child.strip()
                        if not child:
                            continue
                        is_table = "[TABLE]" in child or ("|" in child and "---" in child)
                        chunk_type = "toc" if page_is_toc else ("table" if is_table else "text")
                        output.append({
                            "id": str(uuid.uuid4()),
                            "document_id": document_id,
                            "content": child,
                            "chunk_index": global_index,
                            "page_number": page_number,
                            "page_end": page_number,
                            "section": section_title,
                            "parent_id": parent_id,
                            "parent_content": parent_text,
                            "chunk_type": chunk_type,
                            "char_start": char_offset,
                            "char_end": char_offset + len(child),
                            "token_count": self._estimate_tokens(child),
                            "document_filename": document_filename,
                            "metadata": {
                                "extraction_source": page.get("source", "native"),
                                "heading_path": section["heading_path"],
                                "heading_level": section["level"],
                                "chapter": (
                                    section["heading_path"][0]
                                    if section["heading_path"] else None
                                ),
                                "document_filename": document_filename,
                                "is_toc": page_is_toc,
                                "structure_version": 2,
                            },
                            "origin_sig": "quinc-fptu-cc-by-nc-4.0",
                        })
                        global_index += 1
                        char_offset += len(child)
        logger.info("Page-aware chunked document %s -> %d child chunks", document_id, len(output))
        return output

    # ─── Internal Structural & Semantic Chunking Logic ───────────────────────

    @staticmethod
    def is_toc_text(text: str) -> bool:
        """Detect a table-of-contents block without treating it as normal prose."""
        sample = (text or "")[:12000]
        if re.search(r"(?im)^\s*(table\s+of\s+contents|contents|mục\s+lục)\s*$", sample):
            return True
        lines = [line.strip() for line in sample.splitlines() if line.strip()]
        leader_lines = sum(
            bool(re.search(r"(?:\.{4,}|·{4,}|…{2,})\s*\d{1,4}\s*$", line))
            for line in lines
        )
        numbered_entries = sum(
            bool(re.search(r"(?:chapter|chương|section|phần)\s+\d+.*\s\d{1,4}\s*$", line, re.I))
            for line in lines
        )
        return leader_lines >= 4 or numbered_entries >= 6

    @staticmethod
    def _heading_info(line: str) -> tuple[int, str] | None:
        """Return a conservative heading level/title, avoiding code and prose lines."""
        value = line.strip()
        if not value or len(value) > 120:
            return None
        if re.fullmatch(r"\[[A-Z_ -]+\]", value):
            return None
        if re.match(
            r"^(response|output|input|result|example|format instructions|prompt)\b",
            value,
            re.I,
        ):
            return None
        if re.search(r"[={}<>]|\b(?:def|class|return|import|print)\s*\(?", value):
            return None
        markdown = re.match(r"^(#{1,6})\s+(.+?)\s*$", value)
        if markdown:
            return len(markdown.group(1)), markdown.group(2).strip()
        chapter = re.match(
            r"^(chapter|chương|part|phần|appendix|phụ\s+lục)\s+([\w.-]+)\s*[:.-]?\s*(.*)$",
            value,
            re.I,
        )
        if chapter:
            title = " ".join(part for part in chapter.groups() if part).strip()
            return (1 if chapter.group(1).casefold() in {"part", "phần"} else 2), title
        numbered = re.match(r"^(\d+(?:\.\d+){0,4})[.)]?\s+(.+)$", value)
        if numbered and len(numbered.group(2).split()) <= 14 and not value.endswith(('.', ';', ',')):
            return min(6, numbered.group(1).count(".") + 2), value
        letters = [char for char in value if char.isalpha()]
        if (
            3 <= len(letters)
            and len(value.split()) <= 10
            and all(char.isupper() for char in letters)
            and not value.endswith((".", ";", ","))
        ):
            return 2, value
        if (
            value.endswith(":")
            and len(value.split()) <= 12
            and re.fullmatch(r"[A-Z0-9][A-Z0-9\s&/().,'’\-]+:", value)
        ):
            return 3, value[:-1].strip()
        return None

    def _split_structured_sections(
        self, text: str, inherited_path: List[str] | None = None
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Split text while retaining a stable hierarchical heading path."""
        path = list(inherited_path or [])
        sections: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        current_path = list(path)
        current_level = len(path) or 0

        def flush():
            if not any(line.strip() for line in current_lines):
                return
            sections.append({
                "title": current_path[-1] if current_path else "",
                "heading_path": list(current_path),
                "level": current_level,
                "content": "\n".join(current_lines),
                "ordinal": len(sections),
            })

        for line in text.split("\n"):
            heading = self._heading_info(line)
            if heading:
                flush()
                current_lines = []
                level, title = heading
                path = path[: max(0, level - 1)]
                path.append(title)
                current_path = list(path)
                current_level = level
            current_lines.append(line)
        flush()
        if not sections:
            sections = [{
                "title": path[-1] if path else "",
                "heading_path": list(path),
                "level": len(path),
                "content": text,
                "ordinal": 0,
            }]
        return sections, path

    def _split_by_sections(self, text: str) -> List[tuple]:
        """
        Split text by Markdown headings (#, ##, ###) or major structural breaks.
        Returns a list of (section_title, section_content) tuples.
        """
        sections, _ = self._split_structured_sections(text)
        return [(item["title"], item["content"]) for item in sections]

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
        heading = self._heading_info(first_line)
        return heading[1][:100] if heading else ""

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

