"""
Text Cleaner Service
Preprocessing pipeline for document text before chunking:
- Header/footer detection & removal
- Date normalization
- Noise removal (page numbers, watermarks, repeated disclaimers)
- Whitespace normalization
"""
import re
import logging
from typing import List, Optional
from collections import Counter

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Document text preprocessing pipeline.
    Cleans raw extracted text to improve chunking and entity extraction quality.
    """

    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_dates: bool = True,
        remove_noise: bool = True,
        min_line_length: int = 3,
    ):
        self.remove_headers_footers = remove_headers_footers
        self.normalize_dates = normalize_dates
        self.remove_noise = remove_noise
        self.min_line_length = min_line_length

    def clean(self, text: str, pages: Optional[List[str]] = None) -> str:
        """
        Full cleaning pipeline.

        Args:
            text: Raw document text.
            pages: Optional list of per-page texts for header/footer detection.
        """
        if pages and self.remove_headers_footers:
            text = self._remove_repeated_headers_footers(text, pages)

        if self.remove_noise:
            text = self._remove_page_numbers(text)
            text = self._remove_watermarks(text)
            text = self._remove_empty_lines_noise(text)

        if self.normalize_dates:
            text = self._normalize_dates(text)

        text = self._normalize_whitespace(text)
        text = self._remove_non_printable(text)

        return text.strip()

    # ─── Header / Footer Detection ───────────────────────────────────────────

    def _remove_repeated_headers_footers(
        self, text: str, pages: List[str], threshold: float = 0.6
    ) -> str:
        """
        Detect lines that repeat across >threshold of pages (headers/footers)
        and remove them from the full text.
        """
        if len(pages) < 3:
            return text

        # Collect first 3 and last 3 lines of each page
        candidate_lines: Counter = Counter()
        for page_text in pages:
            lines = [l.strip() for l in page_text.strip().split("\n") if l.strip()]
            header_candidates = lines[:3]
            footer_candidates = lines[-3:] if len(lines) > 3 else []
            for line in header_candidates + footer_candidates:
                # Normalize for comparison (remove page-specific numbers)
                normalized = re.sub(r"\d+", "N", line.strip())
                if len(normalized) >= self.min_line_length:
                    candidate_lines[normalized] += 1

        # Find lines appearing in > threshold of pages
        repeated = {
            pattern
            for pattern, count in candidate_lines.items()
            if count / len(pages) >= threshold
        }

        if not repeated:
            return text

        # Remove matching lines from full text
        cleaned_lines = []
        for line in text.split("\n"):
            normalized = re.sub(r"\d+", "N", line.strip())
            if normalized not in repeated:
                cleaned_lines.append(line)

        removed_count = len(text.split("\n")) - len(cleaned_lines)
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} repeated header/footer lines")

        return "\n".join(cleaned_lines)

    # ─── Page Number Removal ─────────────────────────────────────────────────

    def _remove_page_numbers(self, text: str) -> str:
        """Remove standalone page number lines like 'Page 3', '- 12 -', '3/15'."""
        patterns = [
            r"^[\s]*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$",           # "- 3 -", "12"
            r"^[\s]*[Pp]age\s+\d{1,4}\s*(of\s+\d{1,4})?\s*$",  # "Page 3 of 15"
            r"^[\s]*\d{1,4}\s*/\s*\d{1,4}\s*$",                 # "3/15"
            r"^[\s]*Trang\s+\d{1,4}\s*$",                       # Vietnamese: "Trang 3"
        ]
        combined = "|".join(f"({p})" for p in patterns)
        return re.sub(combined, "", text, flags=re.MULTILINE)

    # ─── Watermark Removal ───────────────────────────────────────────────────

    def _remove_watermarks(self, text: str) -> str:
        """Remove common watermark patterns."""
        watermark_patterns = [
            r"(?i)^[\s]*(?:confidential|draft|internal use only|do not distribute)\s*$",
            r"(?i)^[\s]*(?:bản nháp|nội bộ|mật)\s*$",
        ]
        for pattern in watermark_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        return text

    # ─── Date Normalization ──────────────────────────────────────────────────

    def _normalize_dates(self, text: str) -> str:
        """Normalize common date formats to ISO-like format."""
        # DD/MM/YYYY → YYYY-MM-DD
        text = re.sub(
            r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b",
            lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}",
            text,
        )
        return text

    # ─── Noise Removal ───────────────────────────────────────────────────────

    def _remove_empty_lines_noise(self, text: str) -> str:
        """Remove excessive empty lines and very short noisy lines."""
        lines = text.split("\n")
        cleaned = []
        empty_count = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                empty_count += 1
                if empty_count <= 2:
                    cleaned.append("")
                continue
            empty_count = 0

            # Skip very short lines that are likely noise (but keep list items)
            if (
                len(stripped) < self.min_line_length
                and not re.match(r"^[-•*]\s", stripped)
                and not re.match(r"^\d+\.", stripped)
                and not re.match(r"^#{1,6}\s", stripped)
            ):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    # ─── Whitespace Normalization ────────────────────────────────────────────

    def _normalize_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        text = re.sub(r"\t", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _remove_non_printable(self, text: str) -> str:
        """Remove null bytes and non-printable chars (keep unicode)."""
        text = re.sub(r"\x00", "", text)
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text
