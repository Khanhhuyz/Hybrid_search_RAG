# Context Snapshot

## Current State & Architecture
- **Tech Stack**: Next.js 15 (React 19) frontend (port 3000), FastAPI backend (port 8000), local sqlite3 db (`data/grag.db`), Qdrant Vector Storage (supports Qdrant Cloud or local `data/qdrant_storage`), and NetworkX Knowledge Graph (`data/knowledge_graph.json`).
- **Inference**: Ollama running locally (nomic-embed-text for embeddings, llama3.2 for RAG chat and graph construction).
- **Resilience & Fault Tolerance**: Lifespan startup handles service degradation gracefully (`try/except`). `/health` check uses `asyncio.wait_for(timeout=3.0)` to prevent blocking requests.
- **Async Execution & Batching**: All Qdrant Client calls are offloaded via `asyncio.to_thread`. Vector upserts are chunked in batches of 100 points to prevent payload spikes.
- **Upload Safety**: Document uploads use streaming 1MB chunk reading to validate `MAX_FILE_SIZE_MB` without memory exhaustion.
- **Testing & CI/CD**: Unit tests in `backend/tests/` (`test_chunker.py`, `test_rrf.py`). Automated CI pipeline via GitHub Actions (`.github/workflows/ci.yml`).
- **Dependencies**: Fully pinned version bounds in `backend/requirements.txt` and linter configuration in `pyproject.toml`.

## Accomplished Work
- Pinned all Python dependencies in `backend/requirements.txt`.
- Added fault-tolerant startup lifespan and timed-out health check in `backend/app/main.py`.
- Implemented DoS-safe 1MB chunk streaming file upload validation in `backend/app/api/documents.py`.
- Offloaded Qdrant synchronous SDK operations to `asyncio.to_thread` with 100-point batching in `backend/app/services/vector_store.py`.
- Created unit tests (`backend/tests/test_chunker.py`, `backend/tests/test_rrf.py`) using Python standard `unittest`.
- Created GitHub Actions CI workflow (`.github/workflows/ci.yml`) and linter config (`backend/pyproject.toml`).
- Updated `README.md`, `CONTRIBUTING.md`, and `HANDOVER.md` for AI Agent handoffs and collaborator alignment.
- All 5 unit tests passing (`python -m unittest discover -s tests`). Pushed to GitHub `https://github.com/quinc-fptu/mini-graphrag`.

## Next Steps for Collaborators / Next Agents
- **Run Unit Tests**: `cd backend && python -m unittest discover -s tests`
- **Run Native**: Start FastAPI backend (`python -m uvicorn app.main:app`) and Next.js frontend (`npm run dev`).
- **Vibe Coding Rule**: Keep dev servers external to avoid file watcher lock collisions during multi-file AI edits.

