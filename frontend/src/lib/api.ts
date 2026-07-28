/**
 * GRAG API Client
 * Type-safe HTTP client for all backend endpoints.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ProcessingStatus = "pending" | "processing" | "completed" | "failed";

export interface Document {
  id: string;
  filename: string;
  original_name: string;
  file_type: string;
  file_size: number;
  status: ProcessingStatus;
  chunk_count: number;
  entity_count: number;
  error_message?: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentList {
  documents: Document[];
  total: number;
}

export interface ChunkResult {
  id: string;
  document_id: string;
  content: string;
  chunk_index: number;
  page_number?: number;
  section?: string;
  score?: number;
}

export interface SearchResult {
  chunk: ChunkResult;
  score: number;
  document_filename: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
  search_type: string;
}

export interface Citation {
  document_id: string;
  document_filename: string;
  chunk_id: string;
  chunk_index: number;
  page_number?: number;
  relevance_score: number;
  excerpt: string;
}

export interface GraphContext {
  entities: string[];
  relations: { text: string }[];
}

export interface ChatResponse {
  question: string;
  answer: string;
  citations: Citation[];
  graph_context?: GraphContext;
  semantic_chunks_used: number;
  graph_nodes_used: number;
  model_used: string;
  retrieval_mode: "semantic" | "graph" | "hybrid";
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  document_ids: string[];
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
  document_ids: string[];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  total_edges: number;
}

export interface GraphStats {
  nodes: number;
  edges: number;
  entity_types: Record<string, number>;
}

export interface HealthStatus {
  status: string;
  version: string;
  services: {
    ollama: { status: string; model?: string };
    qdrant: { status: string };
  };
}

// ─── HTTP Helper ──────────────────────────────────────────────────────────────

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }

  // Handle 204 No Content
  if (res.status === 204) return undefined as T;

  return res.json();
}

// ─── Documents API ────────────────────────────────────────────────────────────

export const documentsApi = {
  upload: async (file: File): Promise<Document> => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Upload failed: HTTP ${res.status}`);
    }
    return res.json();
  },

  list: (skip = 0, limit = 20): Promise<DocumentList> =>
    request(`/documents/?skip=${skip}&limit=${limit}`),

  get: (id: string): Promise<Document> =>
    request(`/documents/${id}`),

  delete: (id: string): Promise<void> =>
    request(`/documents/${id}`, { method: "DELETE" }),

  status: (id: string): Promise<Pick<Document, "id" | "status" | "chunk_count" | "entity_count" | "error_message">> =>
    request(`/documents/${id}/status`),
};

// ─── Search API ───────────────────────────────────────────────────────────────

export const searchApi = {
  semantic: (query: string, topK = 5, documentIds?: string[]): Promise<SearchResponse> =>
    request("/search/semantic", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK, document_ids: documentIds }),
    }),

  graph: (query: string, topK = 5): Promise<SearchResponse> =>
    request("/search/graph", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK }),
    }),
};

// ─── Chat API ─────────────────────────────────────────────────────────────────

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export const chatApi = {
  ask: (
    question: string,
    history: ChatMessage[] = [],
    useGraph = true,
    topK = 5,
    documentIds?: string[]
  ): Promise<ChatResponse> =>
    request("/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        history,
        use_graph: useGraph,
        top_k: topK,
        document_ids: documentIds,
      }),
    }),
};

// ─── Graph API ────────────────────────────────────────────────────────────────

export const graphApi = {
  visualize: (documentId?: string, maxNodes = 200): Promise<GraphData> =>
    request(`/graph/visualize?max_nodes=${maxNodes}${documentId ? `&document_id=${documentId}` : ""}`),

  queryEntity: (entityName: string, depth = 2): Promise<GraphData> =>
    request("/graph/entity/query", {
      method: "POST",
      body: JSON.stringify({ entity_name: entityName, depth }),
    }),

  stats: (): Promise<GraphStats> =>
    request("/graph/stats"),

  entities: (entityType?: string, limit = 50): Promise<{ entities: unknown[]; total: number }> =>
    request(`/graph/entities?limit=${limit}${entityType ? `&entity_type=${entityType}` : ""}`),

  relationships: (relationType?: string, limit = 50): Promise<{ relationships: unknown[]; total: number }> =>
    request(`/graph/relationships?limit=${limit}${relationType ? `&relation_type=${relationType}` : ""}`),
};

// ─── Health API ───────────────────────────────────────────────────────────────

export const healthApi = {
  check: (): Promise<HealthStatus> =>
    fetch(`${process.env.NEXT_PUBLIC_API_URL?.replace("/api/v1", "") || "http://localhost:8000"}/health`)
      .then((r) => r.json()),
};
