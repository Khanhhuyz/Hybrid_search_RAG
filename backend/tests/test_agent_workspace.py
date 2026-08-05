import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.agent.artifacts import ArtifactWriter, markdown_content, safe_name
from app.services.agent.orchestrator import AgentOrchestrator


class AgentPlannerTests(unittest.TestCase):
    def test_vietnamese_artifact_intents(self):
        self.assertEqual(
            AgentOrchestrator.classify("Tạo bảng so sánh A và B")[1], "table"
        )
        self.assertEqual(AgentOrchestrator.classify("Lập kế hoạch 12 tuần")[1], "plan")
        self.assertEqual(
            AgentOrchestrator.classify("Viết báo cáo PDF")[1], "document"
        )
        self.assertEqual(AgentOrchestrator.classify("Tạo biểu đồ")[1], "chart")

    def test_output_format_selection(self):
        self.assertEqual(
            AgentOrchestrator.choose_format("xuất PDF", "document"), "pdf"
        )
        self.assertEqual(AgentOrchestrator.choose_format("tạo bảng", "table"), "xlsx")

    def test_duplicate_evidence_is_collapsed(self):
        citations = [
            {
                "document_filename": "a.pdf",
                "page_number": 1,
                "excerpt": "same  evidence",
            },
            {"document_filename": "", "page_number": 1, "excerpt": "same evidence"},
        ]
        self.assertEqual(len(AgentOrchestrator._dedupe_citations(citations)), 1)

    def test_table_shape_drift_is_repaired(self):
        table = AgentOrchestrator._normalize_table(
            {"headers": ["A", "B"], "rows": [[1], [2, 3, 4], "single"]}
        )
        self.assertEqual(table["rows"], [["1", ""], ["2", "3 | 4"], ["single", ""]])

    def test_headers_are_inferred_when_missing(self):
        table = AgentOrchestrator._normalize_table({"rows": [[1, 2, 3]]})
        self.assertEqual(table["headers"], ["Column 1", "Column 2", "Column 3"])

    def test_empty_key_objects_are_unwrapped_and_paired(self):
        table = AgentOrchestrator._normalize_table(
            {
                "headers": ["Phương pháp", "Mô tả"],
                "rows": [
                    {"": "Chunk text()"},
                    {"": "Chia nội dung theo độ dài."},
                    {"": "Vector index"},
                    {"": "Chia dữ liệu theo đặc trưng vector."},
                ],
            }
        )
        self.assertEqual(
            table["rows"],
            [
                ["Chunk text()", "Chia nội dung theo độ dài."],
                ["Vector index", "Chia dữ liệu theo đặc trưng vector."],
            ],
        )

    def test_header_keyed_objects_are_ordered(self):
        table = AgentOrchestrator._normalize_table(
            {
                "headers": ["Method", "Description"],
                "rows": [{"Description": "Meaning", "Method": "Recursive"}],
            }
        )
        self.assertEqual(table["rows"], [["Recursive", "Meaning"]])

    def test_comparison_requires_non_empty_descriptions(self):
        valid = {
            "headers": ["Phương pháp", "Mô tả"],
            "rows": [["Recursive", "Theo cấu trúc"], ["Semantic", "Theo ngữ nghĩa"]],
        }
        invalid = {
            "headers": ["Phương pháp", "Recursive"],
            "rows": [["Semantic", ""]],
        }
        self.assertTrue(AgentOrchestrator._valid_comparison_table(valid))
        self.assertFalse(AgentOrchestrator._valid_comparison_table(invalid))

    def test_labeled_answer_becomes_comparison_table(self):
        answer = """Phương pháp 1: Recursive chunking
- Mô tả: Chia văn bản theo cấu trúc.
- Ưu điểm: Giữ ngữ cảnh.
- Hạn chế: Cần cấu hình.
- Trường hợp sử dụng: Tài liệu dài.

Phương pháp 2: Semantic chunking
- Mô tả: Chia theo ngữ nghĩa.
- Ưu điểm: Đoạn văn nhất quán.
- Hạn chế: Tốn embedding.
- Trường hợp sử dụng: Truy hồi chính xác."""
        data = AgentOrchestrator._comparison_from_answer(answer, ["source.pdf"])
        self.assertIsNotNone(data)
        self.assertEqual(len(data["table"]["rows"]), 2)
        self.assertEqual(data["table"]["rows"][1][1], "Chia theo ngữ nghĩa.")


class ArtifactTests(unittest.TestCase):
    DATA = {
        "title": "Comparison",
        "summary": "Grounded summary",
        "table": {"headers": ["A", "B"], "rows": [[1, 2], [3, 4]]},
        "citations": ["source.pdf"],
    }

    def test_safe_filename(self):
        self.assertNotIn("..", safe_name("../../ unsafe report"))

    def test_markdown_has_table_and_sources(self):
        content = markdown_content(self.DATA, "table")
        self.assertIn("| A | B |", content)
        self.assertIn("[S1] source.pdf", content)

    def test_csv_xlsx_pdf_and_svg_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "app.services.agent.artifacts.settings.OUTPUT_DIR", Path(directory)
            ):
                writer = ArtifactWriter()
                for output_format in ("csv", "xlsx", "pdf", "svg"):
                    artifact = writer.write("run", "table", self.DATA, output_format)
                    self.assertGreater(artifact["size"], 0)
