"""Offline evaluation endpoints for labelled GraphRAG datasets."""
from fastapi import APIRouter

from app.schemas import EvaluationRequest, EvaluationResponse
from app.services.evaluator import Evaluator

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.post("/run", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """Compute retrieval and answer metrics for pre-generated labelled cases."""
    cases = [case.model_dump() for case in request.cases]
    return Evaluator.evaluate_batch(cases)
