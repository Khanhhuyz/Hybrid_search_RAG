# Context Snapshot

## Current State & Architecture
- **Tech Stack**: Next.js 15 (Turbopack) frontend (port 3000), FastAPI backend (port 8000), local sqlite3 db (`data/grag.db`), Qdrant Vector Storage (supports both Qdrant Cloud Cluster via `QDRANT_URL` + `QDRANT_API_KEY` or local file-based `data/qdrant_storage`), and NetworkX Knowledge Graph (`data/knowledge_graph.json`).
- **Inference**: Ollama running locally (nomic-embed-text for embeddings, llama3.2 for RAG chat and graph construction).
- **Vector Storage**: Connected to Qdrant Cloud cluster (`australia-southeast1` GCP), keeping local machine storage and memory usage 100% lightweight.
- **Streaming & SSE**: Backend supports Server-Sent Events (`/api/v1/chat/stream`) yielding live token streams and metadata events to frontend.
- **RAG Ranking & Traversal**: Reciprocal Rank Fusion (RRF) re-ranks vector candidates and graph nodes. Multi-hop (depth=2) graph traversal expands context.
- **Text Chunking**: Intelligent Structural & Semantic Chunking (`chunker.py`) preserving Markdown headings and section context across sub-chunks.
- **Chat Persistence**: Chat history automatically saved to `localStorage` with a Clear history button.
- **Async Execution**: CPU-bound document extraction and chunking are offloaded via `asyncio.to_thread` to maintain event loop responsiveness.
- **Containerization**: Standardized `Dockerfile` for Backend, `Dockerfile` for Frontend, and `docker-compose.yml` for single-command stack deployment.

## Accomplished Work
- Integrated Qdrant Cloud connection without code edits (`backend/.env`).
- Implemented Intelligent Structural & Semantic Text Chunking (`chunker.py`).
- Implemented real-time SSE token streaming from Ollama to Next.js frontend UI (`chatApi.streamAsk`).
- Implemented Reciprocal Rank Fusion (RRF) scoring algorithm in `RAGPipeline`.
- Enhanced Graph visualization with live entity search, type filters, and highlight controls.
- Added Docker containerization files (`backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`).
- Successfully compiled frontend (`npm run build`) and pushed changes to GitHub repository `https://github.com/quinc-fptu/mini-graphrag`.

## Next Steps for Collaborators / Next Agents
- To run with Qdrant Cloud: Add `QDRANT_URL` and `QDRANT_API_KEY` in `backend/.env`.
- To run with Docker: Run `docker compose up -d` in workspace root.
- To run Native: Run FastAPI backend (`uvicorn app.main:app`) and Next.js frontend (`npm run dev`).
