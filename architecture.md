# GRAG — Mini GraphRAG Architecture

## 1. System Overview

```mermaid
graph TB
    subgraph Client["🖥️ Client Layer"]
        Browser["Web Browser"]
    end

    subgraph Frontend["⚛️ Frontend — Next.js 15 (port 3000)"]
        direction TB
        Pages["Pages"]
        Components["UI Components"]
        ApiClient["API Client (api.ts)"]
        LocalStorage["localStorage (Chat History)"]
    end

    subgraph Backend["🐍 Backend — FastAPI (port 8000)"]
        direction TB
        API["API Layer (Routers)"]
        Services["Service Layer"]
        DB["Database Layer"]
    end

    subgraph External["🔧 External Services"]
        Ollama["Ollama (port 11434)"]
        Qdrant["Qdrant Vector DB (port 6333)"]
    end

    Browser --> Frontend
    Frontend -->|HTTP / SSE| Backend
    Backend -->|Embeddings & LLM| Ollama
    Backend -->|Vector Search| Qdrant
```

---

## 2. Frontend Architecture

```mermaid
graph LR
    subgraph Pages["📄 Pages (Next.js App Router)"]
        Dashboard["/ — Dashboard\n(page.tsx)"]
        Chat["/chat — Chat\n(SSE Streaming)"]
        Search["/search — Semantic Search"]
        Graph["/graph — Graph Visualizer"]
    end

    subgraph Components["🧩 Components"]
        DocDrawer["DocumentDrawer\n(Slide-over panel)"]
        GraphViz["GraphVisualizer\n(D3.js force graph)"]
        Layout["Sidebar / TopNav"]
    end

    subgraph Lib["📚 API Client (lib/api.ts)"]
        DocsAPI["documentsApi\n(upload, list, delete, chunks)"]
        SearchAPI["searchApi\n(semantic, graph)"]
        ChatAPI["chatApi\n(ask, streamAsk via SSE)"]
        GraphAPI["graphApi\n(visualize, queryEntity, stats)"]
        HealthAPI["healthApi\n(check)"]
    end

    Dashboard --> DocsAPI
    Chat --> ChatAPI
    Search --> SearchAPI
    Graph --> GraphAPI
    Dashboard --> HealthAPI
    Chat --> DocDrawer
    Graph --> GraphViz
```

---

## 3. Backend Architecture

```mermaid
graph TB
    subgraph API["🌐 API Layer (FastAPI Routers)"]
        DocRouter["/api/v1/documents\n• POST /upload\n• GET /\n• GET /{id}\n• DELETE /{id}\n• GET /{id}/chunks\n• GET /{id}/status"]
        SearchRouter["/api/v1/search\n• POST /semantic\n• POST /graph"]
        ChatRouter["/api/v1/chat\n• POST /\n• POST /stream (SSE)"]
        GraphRouter["/api/v1/graph\n• GET /visualize\n• POST /entity/query\n• GET /stats\n• GET /entities\n• GET /relationships"]
        Health["/health"]
    end

    subgraph DI["🔌 Dependency Injection"]
        Dependencies["dependencies.py\n(Singleton Services)"]
    end

    subgraph Services["⚙️ Service Layer"]
        DocProcessor["DocumentProcessor\n(PDF, DOCX, TXT, MD)"]
        Chunker["TextChunker\n(Header-aware, section context)"]
        Embedder["EmbedderService\n(Ollama nomic-embed-text)"]
        VectorStore["VectorStoreService\n(Qdrant client)"]
        GraphBuilder["GraphBuilderService\n(NetworkX + LLM extraction)"]
        RAGPipeline["RAGPipeline\n(Hybrid search + RRF + LLM)"]
    end

    subgraph Storage["💾 Storage Layer"]
        SQLite["SQLite (aiosqlite)\n• DocumentModel\n• ChunkModel"]
        QdrantDB["Qdrant Vector DB\n(cosine similarity)"]
        GraphJSON["NetworkX Graph\n(knowledge_graph.json)"]
        FileStore["File Storage\n(data/uploads/)"]
    end

    API --> DI
    DI --> Services
    DocRouter --> DocProcessor
    DocRouter --> Chunker
    DocRouter --> Embedder
    DocRouter --> VectorStore
    DocRouter --> GraphBuilder
    SearchRouter --> RAGPipeline
    ChatRouter --> RAGPipeline
    GraphRouter --> GraphBuilder

    DocProcessor --> FileStore
    Chunker --> SQLite
    Embedder --> QdrantDB
    VectorStore --> QdrantDB
    GraphBuilder --> GraphJSON
    RAGPipeline --> VectorStore
    RAGPipeline --> GraphBuilder
    RAGPipeline --> Embedder
```

---

## 4. Document Ingestion Pipeline

```mermaid
flowchart LR
    Upload["📤 Upload\n(PDF/DOCX/TXT/MD)"] --> Save["💾 Save to disk\n+ SQLite metadata"]
    Save --> Extract["📝 DocumentProcessor\nExtract raw text"]
    Extract --> Chunk["✂️ TextChunker\nHeader-aware splitting\n(chunk_size=512, overlap=64)"]
    Chunk --> Parallel{{"⚡ Parallel Processing"}}
    Parallel --> Embed["🧮 EmbedderService\nOllama nomic-embed-text\n→ 768-dim vectors"]
    Parallel --> Entity["🔍 GraphBuilder\nLLM Entity Extraction\n(llama3.2)"]
    Embed --> Upsert["📦 VectorStoreService\nQdrant upsert"]
    Entity --> Graph["🕸️ NetworkX\nAdd nodes & edges\n→ knowledge_graph.json"]
    Upsert --> Done["✅ Status: completed"]
    Graph --> Done

    style Upload fill:#4CAF50,color:#fff
    style Done fill:#2196F3,color:#fff
```

---

## 5. Query & RAG Pipeline (Hybrid GraphRAG)

```mermaid
flowchart TB
    Query["❓ User Question"] --> Translate["🌐 Auto-translate\nto English (if needed)"]
    Translate --> EmbedQ["🧮 Embed Query\n(nomic-embed-text)"]

    EmbedQ --> Semantic["🔍 Semantic Search\nQdrant top-k×2 candidates"]
    Query --> GraphSearch["🕸️ Graph Search\nfind_entities_in_text()"]

    GraphSearch --> MultiHop["🔗 2-hop Graph Traversal\nNetworkX neighborhood"]

    Semantic --> RRF["⚖️ RRF Re-ranking\nReciprocal Rank Fusion\nscore = 1/(k+rank) + 0.5×cosine"]

    RRF --> BuildPrompt["📝 Build RAG Prompt\n• System prompt\n• Semantic context\n• Graph context\n• Question"]
    MultiHop --> BuildPrompt

    BuildPrompt --> Mode{{"Retrieval Mode?"}}
    Mode -->|Semantic + Graph| Hybrid["🔀 hybrid"]
    Mode -->|Semantic only| SemOnly["🔍 semantic"]
    Mode -->|Graph only| GraphOnly["🕸️ graph"]

    Hybrid --> LLM["🤖 Ollama llama3.2\nGenerate answer"]
    SemOnly --> LLM
    GraphOnly --> LLM

    LLM --> Response["📨 Response\n• Answer text\n• Citations\n• Graph context\n• Retrieval mode"]

    LLM --> Stream["📡 SSE Stream\n(token-by-token)"]

    style Query fill:#FF9800,color:#fff
    style RRF fill:#9C27B0,color:#fff
    style LLM fill:#4CAF50,color:#fff
```

---

## 6. Knowledge Graph Structure

```mermaid
graph LR
    subgraph EntityTypes["Entity Types"]
        PERSON["👤 PERSON"]
        ORG["🏢 ORGANIZATION"]
        COURSE["📚 COURSE"]
        DEPT["🏛️ DEPARTMENT"]
        PRODUCT["📦 PRODUCT"]
        LOCATION["📍 LOCATION"]
        CONCEPT["💡 CONCEPT"]
    end

    subgraph RelationTypes["Relation Types"]
        R1["WORKS_AT"]
        R2["BELONGS_TO"]
        R3["HAS_PREREQUISITE"]
        R4["MENTIONS"]
        R5["RELATED_TO"]
        R6["LOCATED_IN"]
    end

    PERSON -->|WORKS_AT| ORG
    COURSE -->|HAS_PREREQUISITE| COURSE
    COURSE -->|BELONGS_TO| DEPT
    PRODUCT -->|RELATED_TO| CONCEPT
    ORG -->|LOCATED_IN| LOCATION
```

---

## 7. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 15, React 19, TypeScript | UI Framework |
| **UI Components** | Lucide React (icons) | Icon library |
| **Graph Viz** | D3.js v7 | Force-directed graph visualization |
| **Styling** | TailwindCSS v4 | Utility-first CSS |
| **Backend** | FastAPI, Uvicorn | REST API + SSE streaming |
| **ORM** | SQLAlchemy 2.0 + aiosqlite | Async SQLite access |
| **Embeddings** | Ollama + nomic-embed-text | 768-dim dense vectors |
| **LLM** | Ollama + llama3.2 (3B) | Text generation & entity extraction |
| **Vector DB** | Qdrant | Cosine similarity search |
| **Knowledge Graph** | NetworkX | In-memory directed graph (JSON persistence) |
| **Doc Parsing** | pdfplumber, python-docx | PDF & DOCX extraction |
| **Containerization** | Docker Compose | Multi-service orchestration |

---

## 8. Docker Deployment Architecture

```mermaid
graph TB
    subgraph Docker["🐳 Docker Compose"]
        subgraph QdrantC["Container: grag-qdrant"]
            QdrantSvc["Qdrant v1.12.4\n:6333 (HTTP)\n:6334 (gRPC)"]
            QdrantVol[("qdrant_storage\n(volume)")]
        end

        subgraph BackendC["Container: grag-backend"]
            BackendSvc["FastAPI\nPython 3.11-slim\n:8000"]
            DataVol[("./data\n(bind mount)")]
        end

        subgraph FrontendC["Container: grag-frontend"]
            FrontendSvc["Next.js 15\n:3000"]
        end
    end

    subgraph Host["🖥️ Host Machine"]
        OllamaSvc["Ollama\n:11434"]
    end

    FrontendSvc -->|depends_on| BackendSvc
    BackendSvc -->|depends_on\nhealthcheck| QdrantSvc
    BackendSvc -->|host.docker.internal\n:11434| OllamaSvc
    BackendSvc --> DataVol
    QdrantSvc --> QdrantVol
```

---

## 9. Data Flow Summary

```mermaid
sequenceDiagram
    actor User
    participant FE as Next.js Frontend
    participant BE as FastAPI Backend
    participant Ollama as Ollama (LLM)
    participant Qdrant as Qdrant (Vector DB)
    participant NX as NetworkX (Graph)

    Note over User,NX: 📤 Document Ingestion Flow
    User->>FE: Upload PDF/DOCX
    FE->>BE: POST /documents/upload
    BE->>BE: Extract text (pdfplumber/docx)
    BE->>BE: Chunk text (header-aware)
    par Parallel Processing
        BE->>Ollama: Embed chunks (nomic-embed-text)
        Ollama-->>BE: 768-dim vectors
        BE->>Qdrant: Upsert vectors + metadata
    and
        BE->>Ollama: Extract entities (llama3.2)
        Ollama-->>BE: Entities & relations JSON
        BE->>NX: Add nodes & edges
    end
    BE-->>FE: Document status: completed

    Note over User,NX: ❓ Query & RAG Flow
    User->>FE: Ask question
    FE->>BE: POST /chat/stream
    BE->>Ollama: Embed question
    Ollama-->>BE: Query vector
    par Hybrid Retrieval
        BE->>Qdrant: Similarity search (top-k)
        Qdrant-->>BE: Ranked chunks
    and
        BE->>NX: Entity lookup + 2-hop traversal
        NX-->>BE: Graph context
    end
    BE->>BE: RRF re-ranking
    BE->>BE: Build RAG prompt
    BE->>Ollama: Generate answer (llama3.2)
    loop SSE Stream
        Ollama-->>BE: Token
        BE-->>FE: data: {token}
        FE-->>User: Display token
    end
```

---

## 10. Evaluation & Maintenance

Labelled cases are evaluated through `POST /api/v1/evaluation/run`. Retrieval
quality uses precision, recall, F1, hit rate, and reciprocal rank; answer overlap
uses Unicode-aware token F1. Runtime latency, confidence, routing distribution,
and errors are available under `/api/v1/monitoring/*`.

```text
Labelled dataset -> offline evaluation -> error analysis
-> tune chunking/schema/retrieval/prompt -> re-index -> compare regression metrics
```
