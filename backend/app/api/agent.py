"""Agentic workspace API: create products, inspect runs, download artifacts."""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.database import ArtifactModel, AsyncSessionLocal
from app.dependencies import get_agent

router = APIRouter(prefix="/agent", tags=["Agent Workspace"])


class AgentRequest(BaseModel):
    request: str = Field(..., min_length=3, max_length=8000)
    document_ids: Optional[list[str]] = None
    output_format: Optional[str] = Field(default=None, pattern="^(md|csv|xlsx|pdf|svg)$")


class ActionApproval(BaseModel):
    action_id: str
    approved: bool
    comment: Optional[str] = None


@router.post("/run")
async def run_agent(payload: AgentRequest, agent=Depends(get_agent)):
    if not settings.AGENT_ENABLED or agent is None:
        raise HTTPException(status_code=503, detail="Agent workspace is disabled")
    try:
        return await agent.run(payload.request, payload.document_ids, payload.output_format)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc


@router.get("/runs")
async def list_runs(limit: int = Query(20, ge=1, le=100), agent=Depends(get_agent)):
    return {"runs": await agent.history(limit)}


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact(artifact_id: str):
    async with AsyncSessionLocal() as db:
        artifact = await db.get(ArtifactModel, artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path, root = Path(artifact.file_path).resolve(), settings.OUTPUT_DIR.resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return FileResponse(path, filename=artifact.filename, media_type=artifact.mime_type)


@router.post("/actions/approve")
async def approve_action(approval: ActionApproval):
    if not approval.approved:
        return {"action_id": approval.action_id, "status": "cancelled"}
    if not settings.ACTIONS_ENABLED:
        return {
            "action_id": approval.action_id,
            "status": "not_configured",
            "message": "External actions are disabled until an integration is configured.",
        }
    raise HTTPException(status_code=501, detail="No external action provider is configured")


@router.get("/capabilities")
async def capabilities():
    return {
        "intents": ["question", "table", "document", "plan", "chart"],
        "formats": ["md", "csv", "xlsx", "pdf", "svg"],
        "actions_enabled": settings.ACTIONS_ENABLED,
        "approval_required": settings.ACTION_REQUIRE_APPROVAL,
    }
