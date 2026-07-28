# Project Handover & Vibe Coding Guide

> **Project Status:** Experimental Side Project / Work In Progress (WIP). Built via AI Pair Programming (Vibe Coding).

This document provides a summary of the project architecture, environment setup, and AI collaboration guidelines for developers and contributors.

---

## 🎨 Vibe Coding Workflow

This codebase was developed iteratively with AI assistance. When continuing development:

1. **Avoid Background Auto-Reloading Servers in AI Tool Calls:**
   Run `npm run dev` and `uvicorn` in separate terminal windows on your local machine to avoid file watcher lock collisions during multi-file edits.
2. **Context Snapshot Alignment:**
   Keep `.agents/doc/context-snapshot.md` updated so AI agents instantly grasp the project state.
3. **Build Pre-Flight:**
   Run `npm run build` inside `frontend/` before pushing commits to ensure clean TypeScript compilation.

---

## System Overview

- **Frontend:** Next.js 15 (React 19, Tailwind CSS, D3.js) on port `3000`.
- **Backend:** FastAPI (Python 3.11+, SQLAlchemy, Pydantic) on port `8000`.
- **Vector DB:** Qdrant Cloud or local disk storage (`data/qdrant_storage`).
- **Graph Store:** NetworkX Graph serialized to `data/knowledge_graph.json`.
- **Database:** SQLite DB at `data/grag.db`.

---

## Setup Instructions

### 1. Local Native Run

```bash
# Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2

# Backend
cd backend
python -m venv .venv
# Activate environment
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000 --reload

# Frontend
cd frontend
npm install
npm run dev
```

### 2. Qdrant Cloud Setup

Add credentials to `backend/.env`:
```ini
QDRANT_URL="https://your-cluster-id.cloud.qdrant.io"
QDRANT_API_KEY="your_api_key_here"
```

To sync local vectors to cloud:
```bash
python scripts/migrate_to_cloud.py
```

### 3. Docker Run

```bash
docker compose up -d
```

---

## Core Components

- `backend/app/services/rag_pipeline.py`: RAG logic, RRF reranking, and SSE stream generator.
- `backend/app/services/chunker.py`: Heading-based document chunking.
- `backend/app/api/search.py`: SSE chat streaming endpoint.
- `frontend/src/components/documents/DocumentDrawer.tsx`: Slide-over drawer for chunk viewing and citation highlighting.
- `frontend/src/components/graph/GraphVisualizer.tsx`: D3 graph visualizer.
- `AGENTS.md`: Environment stability guidelines.

---

## Repository

- **GitHub:** [`https://github.com/quinc-fptu/mini-graphrag`](https://github.com/quinc-fptu/mini-graphrag)
