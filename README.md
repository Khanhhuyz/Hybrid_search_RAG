# GRAG — Mini Semantic Search + GraphRAG System

> **G**raph **R**etrieval-**A**ugmented **G**eneration — Upload documents, ask questions, visualize knowledge.

![Architecture](docs/architecture.png)

## Features

- 📄 **Multi-format Document Upload** — PDF, DOCX, TXT, Markdown
- 🔍 **Semantic Search** — Vector similarity via Qdrant + Ollama embeddings
- 🧠 **Knowledge Graph** — Auto-extracted entities & relationships via LLM
- 🔗 **GraphRAG** — Hybrid retrieval (vector + graph) for richer answers
- 💬 **Chat Interface** — Ask questions with citation support
- 📊 **Graph Visualization** — Interactive D3.js graph viewer
- ⚡ **100% Local** — Powered by Ollama (no API keys required)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14 + TypeScript + Tailwind CSS |
| Backend | FastAPI (Python 3.11+) |
| Embeddings | Ollama (`nomic-embed-text`) |
| LLM | Ollama (`llama3.2`) |
| Vector DB | Qdrant |
| Graph DB | NetworkX + JSON |
| Metadata DB | SQLite |

---

## Prerequisites

1. **Python 3.11+** — `python --version`
2. **Node.js 18+** — `node --version`
3. **Docker Desktop** — for Qdrant
4. **Ollama** — [Install](https://ollama.ai)

---

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d qdrant
```

### 2. Pull Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 3. Backend Setup

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Backend runs at: http://localhost:8000  
API docs: http://localhost:8000/docs

### 4. Frontend Setup

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:3000

---

## API Reference

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload a document |
| GET | `/api/v1/documents/` | List all documents |
| GET | `/api/v1/documents/{id}` | Get document details |
| DELETE | `/api/v1/documents/{id}` | Delete document |
| GET | `/api/v1/documents/{id}/status` | Processing status |

### Search & Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/search/semantic` | Semantic vector search |
| POST | `/api/v1/search/graph` | Knowledge graph search |
| POST | `/api/v1/chat` | GraphRAG chat with citations |

### Graph

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/graph/visualize` | Full graph for visualization |
| POST | `/api/v1/graph/entity/query` | Query entity neighborhood |
| GET | `/api/v1/graph/entities` | List all entities |
| GET | `/api/v1/graph/relationships` | List all relationships |
| GET | `/api/v1/graph/stats` | Graph statistics |

---

## Project Structure

```
GRAG/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app
│   │   ├── config.py             # Settings
│   │   ├── database.py           # SQLite models
│   │   ├── schemas.py            # Pydantic schemas
│   │   ├── dependencies.py       # DI container
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   ├── search.py
│   │   │   └── graph.py
│   │   └── services/
│   │       ├── document_processor.py
│   │       ├── chunker.py
│   │       ├── embedder.py
│   │       ├── vector_store.py
│   │       ├── graph_builder.py
│   │       └── rag_pipeline.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx          # Dashboard
│       │   ├── chat/page.tsx     # Chat interface
│       │   └── graph/page.tsx    # Graph visualization
│       ├── components/
│       └── lib/
├── data/                         # Auto-created
│   ├── uploads/
│   ├── grag.db
│   └── knowledge_graph.json
├── docker-compose.yml
└── README.md
```

---

## Configuration

Edit `backend/.env` to customize:

```bash
EMBEDDING_MODEL=nomic-embed-text   # or mxbai-embed-large
LLM_MODEL=llama3.2                 # or qwen2.5, mistral, etc.
CHUNK_SIZE=512
CHUNK_OVERLAP=64
SIMILARITY_THRESHOLD=0.6
```

---

## License

MIT
