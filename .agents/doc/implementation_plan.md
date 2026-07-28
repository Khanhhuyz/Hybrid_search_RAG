# Implementation Plan

## Phase 1: Stability & Bug Fixes (Completed)
- [x] Fix Qdrant API call to use `query_points` with correct `score_threshold` filtering behavior.
- [x] Fix SQLite database lock errors by adjusting commit timings and adding database connection timeout (30s).
- [x] Implement parallel entity extraction (`asyncio.Semaphore(3)`) to speed up processing.
- [x] Increase max file size limit to 500MB on both Frontend and Backend.

## Phase 2: User Experience (Completed)
- [x] Design and apply Minimalist Editorial styling for the main Dashboard.
- [x] Add overall metrics dashboard (Total Docs, Chunks, Nodes).
- [x] Add live 4-step status progress stepper (`Read` -> `Chunk` -> `Embed` -> `Graph`) to DocumentCard.
- [x] Add filename, status, and format search/filter bar.

## Phase 3: RAG Optimizations & Streaming (Completed)
- [x] Implement real-time SSE token streaming (`/api/v1/chat/stream`) & frontend streaming UI (`chatApi.streamAsk`).
- [x] Implement Reciprocal Rank Fusion (RRF) re-ranking for vector + graph context fusion.
- [x] Implement Multi-hop (depth=2) graph query expansion.
- [x] Implement Intelligent Structural & Semantic Chunking (`chunker.py`).
- [x] Add local chat history persistence (`localStorage`) with Clear history button.
- [x] Implement automatic query translation to English inside `rag_pipeline.py` using Ollama.

## Phase 4: Containerization & Cloud Integration (Completed)
- [x] Add Qdrant Cloud cluster support via `QDRANT_URL` and `QDRANT_API_KEY` in `backend/.env`.
- [x] Create `backend/Dockerfile` and `frontend/Dockerfile`.
- [x] Standardize full-stack orchestration in `docker-compose.yml`.
- [x] Push all code to Private GitHub Repository (`https://github.com/quinc-fptu/mini-graphrag`).

## Handover Instructions for Next Collaborator / Agent
1. **Running Native:**
   - Backend: `cd backend; python -m uvicorn app.main:app --port 8000 --reload`
   - Frontend: `cd frontend; npm run dev`
2. **Running with Qdrant Cloud:** Set `QDRANT_URL` and `QDRANT_API_KEY` in `backend/.env`.
3. **Running with Docker:** Run `docker compose up -d` in project root.

