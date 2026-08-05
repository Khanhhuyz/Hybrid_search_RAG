"""Bounded Agentic GraphRAG orchestration for creating work products."""

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.config import settings
from app.database import AgentRunModel, ArtifactModel, AsyncSessionLocal

from .artifacts import ArtifactWriter


class AgentOrchestrator:
    def __init__(self, rag):
        self.rag = rag
        self.writer = ArtifactWriter()

    @staticmethod
    def classify(request: str) -> tuple[str, str]:
        text = request.lower()
        if any(word in text for word in ("biểu đồ", "chart", "graph chart")):
            return "create_chart", "chart"
        if any(word in text for word in ("kế hoạch", "roadmap", "project plan")):
            return "create_plan", "plan"
        if any(word in text for word in ("bảng", "table", "so sánh", "compare")):
            return "create_table", "table"
        if any(
            word in text
            for word in ("báo cáo", "report", "proposal", "memo", "tài liệu")
        ):
            return "create_document", "document"
        return "factual_question", "answer"

    @staticmethod
    def choose_format(request: str, artifact_type: str) -> str:
        text = request.lower()
        if "pdf" in text:
            return "pdf"
        if any(word in text for word in ("excel", "xlsx", "spreadsheet")):
            return "xlsx"
        if "csv" in text:
            return "csv"
        if artifact_type == "table":
            return "xlsx"
        if artifact_type == "chart":
            return "svg"
        return "md"

    async def run(self, request: str, document_ids=None, output_format=None) -> dict:
        run_id = str(uuid.uuid4())
        intent, artifact_type = self.classify(request)
        plan = ["Analyze intent", "Retrieve grounded evidence", "Validate citations"]
        if artifact_type != "answer":
            plan += [f"Structure {artifact_type}", "Create and validate artifact"]
        async with AsyncSessionLocal() as db:
            db.add(
                AgentRunModel(
                    id=run_id,
                    request=request,
                    intent=intent,
                    plan_json=json.dumps(plan, ensure_ascii=False),
                    status="running",
                )
            )
            await db.commit()

        steps = []
        try:
            steps.append({"step": "planning", "status": "completed", "detail": intent})
            result = await asyncio.wait_for(
                self.rag.answer(
                    question=request,
                    top_k=8,
                    use_graph=True,
                    document_ids=document_ids,
                    search_type="hybrid",
                ),
                timeout=settings.AGENT_TIMEOUT,
            )
            citations = self._dedupe_citations(result.get("citations", []))
            steps.append(
                {
                    "step": "retrieval",
                    "status": "completed",
                    "evidence": len(citations),
                    "mode": result.get("retrieval_mode"),
                    "groundedness_score": result.get("groundedness_score", 0.0),
                    "confidence_score": result.get("confidence_score", 0.0),
                }
            )
            evidence_status = (
                "sufficient"
                if len(citations) >= 2
                else "partial"
                if citations
                else "insufficient"
            )
            artifacts = []
            answer = result.get("answer", "")
            if artifact_type != "answer":
                structured = await self._structure(
                    request, artifact_type, answer, citations
                )
                output = output_format or self.choose_format(request, artifact_type)
                artifact = self.writer.write(
                    run_id, artifact_type, structured, output
                )
                await self._save_artifact(artifact)
                artifact["download_url"] = (
                    f"{settings.API_PREFIX}/agent/artifacts/{artifact['id']}/download"
                )
                artifact.pop("file_path", None)
                artifacts.append(artifact)
                answer = structured.get("summary") or answer
                steps.append(
                    {
                        "step": "artifact",
                        "status": "completed",
                        "format": output,
                        "artifact_id": artifact["id"],
                    }
                )
            pending_actions = self._pending_actions(request, artifacts)
            await self._finish(run_id, "completed", steps, answer)
            return {
                "run_id": run_id,
                "intent": intent,
                "answer": answer,
                "plan": plan,
                "steps": steps,
                "evidence_status": evidence_status,
                "citations": citations,
                "artifacts": artifacts,
                "pending_actions": pending_actions,
            }
        except Exception as exc:
            await self._finish(run_id, "failed", steps, None, str(exc))
            raise

    async def _structure(self, request, artifact_type, answer, citations):
        sources = [
            f"{citation.get('document_filename', 'Source')} — "
            f"{citation.get('excerpt', '')[:240]}"
            for citation in citations
        ]
        is_comparison = artifact_type == "table" and any(
            word in request.lower() for word in ("so sánh", "compare", "comparison")
        )
        if is_comparison:
            parsed = self._comparison_from_answer(answer, sources)
            if parsed:
                return parsed
        comparison_rules = (
            """
This is a comparison table. table.headers MUST be exactly ["Phương pháp", "Mô tả", "Ưu điểm", "Hạn chế", "Trường hợp sử dụng"]. Put method names ONLY in the first column, never in headers. Every method row MUST have a non-empty description in the second column. Compare only methods explicitly supported by the evidence.
"""
            if is_comparison
            else ""
        )
        base_prompt = f"""Create a grounded {artifact_type} from the supplied answer. Return JSON only.
Schema: {{"title":"...","summary":"...","sections":[{{"heading":"...","content":"... [S1]"}}],"table":{{"headers":["..."],"rows":[["cell 1","cell 2"]]}},"plan":{{"goal":"...","milestones":[{{"name":"...","deadline":"...","tasks":["..."]}}]}},"citations":["..."]}}
Use the same language as the request. Never invent facts or sources. Table rows MUST be arrays, never objects. Every row MUST contain exactly one value per header. For comparison tables, put the item name and its description in the SAME row.
{comparison_rules}
REQUEST: {request[:2000]}
GROUNDED ANSWER: {answer[:10000]}
SOURCES: {json.dumps(sources, ensure_ascii=False)[:8000]}"""
        last_data = {}
        async with httpx.AsyncClient(timeout=settings.AGENT_TIMEOUT) as client:
            for attempt in range(2):
                correction = ""
                if attempt:
                    previous = json.dumps(last_data, ensure_ascii=False)[:5000]
                    correction = (
                        "\nThe previous table was invalid because descriptions were empty "
                        "or method names were used as headers. Rebuild it using the exact "
                        f"comparison schema. PREVIOUS JSON: {previous}"
                    )
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": settings.LLM_MODEL,
                        "prompt": base_prompt + correction,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.05, "num_predict": 1800},
                    },
                )
                response.raise_for_status()
                last_data = json.loads(response.json().get("response", "{}"))
                last_data["citations"] = sources
                last_data["table"] = self._normalize_table(last_data.get("table"))
                if not is_comparison or self._valid_comparison_table(last_data["table"]):
                    return last_data
        raise ValueError(
            "Model could not produce a comparison table with non-empty descriptions"
        )

    @staticmethod
    def _comparison_from_answer(answer, sources):
        """Deterministically convert a labeled grounded answer into comparison rows."""
        headers = [
            "Phương pháp",
            "Mô tả",
            "Ưu điểm",
            "Hạn chế",
            "Trường hợp sử dụng",
        ]
        field_map = {
            "mô tả": 1,
            "description": 1,
            "ưu điểm": 2,
            "advantages": 2,
            "pros": 2,
            "hạn chế": 3,
            "limitations": 3,
            "cons": 3,
            "trường hợp sử dụng": 4,
            "use cases": 4,
            "use case": 4,
        }
        rows, current, active_field = [], None, None
        for raw_line in str(answer or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            method = re.match(
                r"^(?:phương\s*pháp|method)\s*\d*\s*:\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            if method:
                if current:
                    rows.append(current)
                current = [method.group(1).strip(), "", "", "", ""]
                active_field = None
                continue
            if current is None:
                continue
            field = re.match(r"^[-*]\s*([^:]+):\s*(.*)$", line)
            if field:
                label = re.sub(r"\s+", " ", field.group(1).strip().lower())
                active_field = next(
                    (index for name, index in field_map.items() if label == name), None
                )
                if active_field:
                    current[active_field] = field.group(2).strip()
            elif active_field:
                current[active_field] = f"{current[active_field]} {line}".strip()
        if current:
            rows.append(current)
        rows = [row for row in rows if row[0] and row[1]]
        if len(rows) < 2:
            return None
        return {
            "title": "So sánh các phương pháp chunking",
            "summary": (
                f"Bảng tổng hợp {len(rows)} phương pháp chunking được đề cập trong tài liệu."
            ),
            "sections": [],
            "table": {"headers": headers, "rows": rows},
            "plan": {},
            "citations": sources,
        }

    @staticmethod
    def _valid_comparison_table(table):
        headers, rows = table.get("headers", []), table.get("rows", [])
        if len(headers) < 2 or len(rows) < 2:
            return False
        normalized_headers = [
            re.sub(r"[*_]+", "", header).strip().lower() for header in headers
        ]
        has_description = any(
            name in normalized_headers for name in ("mô tả", "description")
        )
        return has_description and all(
            len(row) >= 2 and str(row[0]).strip() and str(row[1]).strip()
            for row in rows
        )

    @staticmethod
    def _normalize_table(table):
        """Repair common LLM table-shape drift without inventing cell content."""
        if not isinstance(table, dict):
            return {"headers": [], "rows": []}
        raw_headers = table.get("headers") or []
        raw_rows = table.get("rows") or []
        headers = (
            [
                str(value).strip() or f"Column {index + 1}"
                for index, value in enumerate(raw_headers)
            ]
            if isinstance(raw_headers, list)
            else []
        )
        rows = []
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if isinstance(row, dict):
                    mapped = (
                        [row.get(header) for header in headers]
                        if headers and any(header in row for header in headers)
                        else list(row.values())
                    )
                    rows.append(mapped)
                elif isinstance(row, (list, tuple)):
                    rows.append(list(row))
                else:
                    rows.append([row])
        if not headers and rows:
            width = max(len(row) for row in rows)
            headers = [f"Column {index + 1}" for index in range(width)]
        width = len(headers)
        if width == 0:
            return {"headers": [], "rows": []}
        if width == 2 and len(rows) >= 2 and all(len(row) == 1 for row in rows):
            rows = [
                rows[index] + (rows[index + 1] if index + 1 < len(rows) else [""])
                for index in range(0, len(rows), 2)
            ]
        normalized = []
        for row in rows:
            values = [AgentOrchestrator._cell_text(value) for value in row]
            if len(values) < width:
                values.extend([""] * (width - len(values)))
            elif len(values) > width:
                values = values[: width - 1] + [" | ".join(values[width - 1 :])]
            normalized.append(values)
        return {"headers": headers, "rows": normalized}

    @staticmethod
    def _cell_text(value):
        if value is None:
            return ""
        if isinstance(value, dict):
            return " · ".join(
                AgentOrchestrator._cell_text(item) for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return ", ".join(AgentOrchestrator._cell_text(item) for item in value)
        return str(value).strip()

    @staticmethod
    def _dedupe_citations(citations):
        """Collapse duplicate Qdrant payloads left by historical re-ingestion."""
        unique, seen = [], set()
        for citation in citations:
            excerpt = re.sub(
                r"\s+", " ", str(citation.get("excerpt", ""))
            ).strip().lower()
            key = (
                citation.get("document_filename") or "",
                citation.get("page_number"),
                excerpt[:180],
            )
            fallback = (citation.get("page_number"), excerpt[:180])
            if key in seen or fallback in seen:
                continue
            seen.add(key)
            seen.add(fallback)
            unique.append(citation)
        return unique

    def _pending_actions(self, request, artifacts):
        text = request.lower()
        action = next(
            (
                name
                for name, words in {
                    "send_email": ("gửi email", "send email"),
                    "create_ticket": ("tạo ticket", "create ticket"),
                    "update_calendar": ("lịch", "calendar"),
                }.items()
                if any(word in text for word in words)
            ),
            None,
        )
        if not action:
            return []
        return [
            {
                "action_id": str(uuid.uuid4()),
                "action_type": action,
                "description": (
                    "External action requires explicit approval and configured integration."
                ),
                "status": "pending_approval",
                "artifact_ids": [artifact["id"] for artifact in artifacts],
            }
        ]

    async def _save_artifact(self, artifact):
        async with AsyncSessionLocal() as db:
            db.add(
                ArtifactModel(
                    id=artifact["id"],
                    run_id=artifact["run_id"],
                    artifact_type=artifact["artifact_type"],
                    title=artifact["title"],
                    filename=artifact["filename"],
                    file_path=artifact["file_path"],
                    mime_type=artifact["mime_type"],
                    size=artifact["size"],
                    preview=artifact["preview"],
                )
            )
            await db.commit()

    async def _finish(self, run_id, status, steps, answer=None, error=None):
        async with AsyncSessionLocal() as db:
            run = await db.get(AgentRunModel, run_id)
            run.status = status
            run.steps_json = json.dumps(steps, ensure_ascii=False)
            run.answer = answer
            run.error_message = error
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()

    async def history(self, limit=20):
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(AgentRunModel)
                .order_by(AgentRunModel.created_at.desc())
                .limit(limit)
            )
            return [
                {
                    "id": run.id,
                    "request": run.request,
                    "intent": run.intent,
                    "status": run.status,
                    "answer": run.answer,
                    "created_at": run.created_at,
                }
                for run in result.scalars().all()
            ]
