/**
 * GRAG API Client (v2)
 * Type-safe HTTP client for all backend endpoints.
 * Supports search_type, confidence scoring, communities, and monitoring.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ProcessingStatus = "pending" | "processing" | "completed" | "failed";
export type SearchType = "local" | "global" | "hybrid" | "auto";

export interface Document {
  id: string;
  filename: string;
  original_name: string;
  file_type: string;
  file_size: number;
  status: ProcessingStatus;
  index_version: number;
  chunk_count: number;
  entity_count: number;
  error_message?: string;
  progress_stage: string;
  progress_current: number;
  progress_total: number;
  heartbeat_at?: string;
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
  retrieval_mode: "semantic" | "graph" | "hybrid" | "global";
  query_type?: string;
  confidence_score?: number;
  confidence_calibrated?: boolean;
  groundedness_score?: number;
  claim_support?: Array<{
    claim: string;
    citations: string[];
    invalid_citations?: string[];
    support_score: number;
    reason?: string | null;
    supported: boolean;
  }>;
  timings_ms?: Record<string, number>;
  warnings?: string[];
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  document_ids: string[];
  properties: Record<string, unknown>;
  community_id?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
  weight: number;
  document_ids: string[];
  description?: string;
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

export interface CommunityReport {
  community_id: number;
  level: number;
  title: string;
  summary: string;
  key_findings: string[];
  main_entities: string[];
  importance_score: number;
}

export interface MonitoringStats {
  total_queries: number;
  error_count: number;
  error_rate: number;
  avg_latency_ms: number;
  avg_confidence: number;
  query_type_distribution: Record<string, number>;
  retrieval_mode_distribution: Record<string, number>;
  stage_latencies: Record<string, { avg_ms: number; min_ms: number; max_ms: number; count: number }>;
  recent_queries_count: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  services: {
    ollama: { status: string; model?: string } | string;
    qdrant: { status: string } | string;
    neo4j: string;
  };
}

export interface AgentArtifact {
  id: string;
  artifact_type: string;
  title: string;
  filename: string;
  mime_type: string;
  size: number;
  preview: string;
  download_url: string;
}

export interface AgentRunResponse {
  run_id: string;
  intent: string;
  answer: string;
  plan: string[];
  steps: Array<Record<string, unknown>>;
  evidence_status: string;
  citations: Citation[];
  artifacts: AgentArtifact[];
  pending_actions: Array<Record<string, unknown>>;
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

  status: (id: string): Promise<Pick<Document, "id" | "status" | "chunk_count" | "entity_count" | "error_message" | "progress_stage" | "progress_current" | "progress_total" | "heartbeat_at">> =>
    request(`/documents/${id}/status`),

  retry: (id: string): Promise<Document> =>
    request(`/documents/${id}/retry`, { method: "POST" }),

  getChunks: (id: string): Promise<{ document_id: string; filename: string; total_chunks: number; chunks: ChunkResult[] }> =>
    request(`/documents/${id}/chunks`),
};

export const agentApi = {
  run: (requestText: string, outputFormat?: string, documentIds?: string[]): Promise<AgentRunResponse> =>
    request("/agent/run", {
      method: "POST",
      body: JSON.stringify({
        request: requestText,
        output_format: outputFormat || null,
        document_ids: documentIds,
      }),
    }),
  history: (): Promise<{ runs: Array<Record<string, unknown>> }> =>
    request("/agent/runs"),
  downloadUrl: (artifactId: string): string =>
    `${API_BASE}/agent/artifacts/${artifactId}/download`,
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
    documentIds?: string[],
    searchType: SearchType = "auto"
  ): Promise<ChatResponse> =>
    request("/chat", {
      method: "POST",
      body: JSON.stringify({
        question,
        history,
        use_graph: useGraph,
        top_k: topK,
        document_ids: documentIds,
        search_type: searchType,
      }),
    }),

  streamAsk: async (
    question: string,
    history: ChatMessage[] = [],
    useGraph = true,
    topK = 5,
    documentIds: string[] | undefined,
    onMetadata: (metadata: Partial<ChatResponse>) => void,
    onToken: (token: string) => void,
    onError: (error: string) => void,
    searchType: SearchType = "auto",
    onDone?: (data: Pick<ChatResponse,
      "confidence_score" | "confidence_calibrated" | "groundedness_score" |
      "claim_support" | "warnings" | "timings_ms"
    >) => void
  ): Promise<void> => {
    const url = `${API_BASE}/chat/stream`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history,
        use_graph: useGraph,
        top_k: topK,
        document_ids: documentIds,
        search_type: searchType,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(err.detail || `Stream failed: HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("Response body reader not available");

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          const jsonStr = trimmed.replace("data: ", "");
          try {
            const parsed = JSON.parse(jsonStr);
            if (parsed.type === "metadata") {
              onMetadata(parsed.data);
            } else if (parsed.type === "token") {
              onToken(parsed.data.text);
            } else if (parsed.type === "done" && onDone) {
              onDone(parsed.data);
            } else if (parsed.type === "error") {
              onError(parsed.data.error);
            }
          } catch {
            // Ignore parse errors
          }
        }
      }
    }
  },
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

  communities: (level?: number, limit = 50): Promise<{ communities: CommunityReport[]; total: number }> =>
    request(`/graph/communities?limit=${limit}${level !== undefined ? `&level=${level}` : ""}`),

  detectCommunities: (): Promise<{ status: string; communities_detected: number; message: string }> =>
    request("/graph/communities/detect", { method: "POST" }),
};

// ─── Monitoring API ───────────────────────────────────────────────────────────

export const monitoringApi = {
  stats: (): Promise<MonitoringStats> =>
    request("/monitoring/stats"),

  recentQueries: (limit = 20): Promise<{ queries: unknown[] }> =>
    request(`/monitoring/queries?limit=${limit}`),
};

// ─── Health API ───────────────────────────────────────────────────────────────

export const healthApi = {
  check: (): Promise<HealthStatus> =>
    fetch(`${process.env.NEXT_PUBLIC_API_URL?.replace("/api/v1", "") || "http://localhost:8000"}/health`)
      .then((r) => r.json()),
};
