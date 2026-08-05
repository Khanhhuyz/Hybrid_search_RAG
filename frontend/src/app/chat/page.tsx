"use client";

import { useState, useRef, useEffect } from "react";
import { chatApi, ChatMessage, ChatResponse, Citation, SearchType } from "@/lib/api";
import { DocumentDrawer } from "@/components/documents/DocumentDrawer";
import {
  Send, Bot, User, BookOpen, Share2, Loader2, Sparkles, AlertCircle, Trash2, ExternalLink,
  Zap, Globe, Target, Shuffle, Shield, Clock,
} from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
  error?: string;
  loading?: boolean;
  confidenceScore?: number;
  timingsMs?: Record<string, number>;
  warnings?: string[];
}

const STORAGE_KEY = "grag_chat_history_v2";

const SEARCH_TYPE_CONFIG: Record<SearchType, { label: string; icon: typeof Target; color: string; desc: string }> = {
  auto: { label: "Auto", icon: Zap, color: "text-amber-400", desc: "Automatic routing" },
  local: { label: "Local", icon: Target, color: "text-emerald-400", desc: "Entity-focused" },
  global: { label: "Global", icon: Globe, color: "text-sky-400", desc: "Map-Reduce summaries" },
  hybrid: { label: "Hybrid", icon: Shuffle, color: "text-violet-400", desc: "Vector + Graph + RRF" },
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [useGraph, setUseGraph] = useState(true);
  const [topK, setTopK] = useState(5);
  const [searchType, setSearchType] = useState<SearchType>("auto");
  const [thinking, setThinking] = useState(false);

  // Drawer state for citation viewing
  const [activeDrawerDoc, setActiveDrawerDoc] = useState<{
    documentId: string;
    filename: string;
    chunkId?: string;
  } | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
          setMessages(JSON.parse(saved));
        }
      } catch {
        // Ignore storage errors
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  // Save chat history to localStorage whenever messages update
  useEffect(() => {
    if (messages.length > 0) {
      try {
        const cleanMessages = messages.filter((m) => !m.loading);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(cleanMessages));
      } catch {
        // Ignore storage errors
      }
    }
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const clearHistory = () => {
    setMessages([]);
    localStorage.removeItem(STORAGE_KEY);
  };

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || thinking) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: question };
    const botMsgId = (Date.now() + 1).toString();
    const botMsg: Message = { id: botMsgId, role: "assistant", content: "", loading: true };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput("");
    setThinking(true);

    // Build history for context
    const history: ChatMessage[] = messages
      .filter((m) => !m.loading && !m.error)
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      await chatApi.streamAsk(
        question,
        history,
        useGraph,
        topK,
        undefined,
        (metadata) => {
          // Received metadata (citations, graph context, retrieval mode)
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botMsgId
                ? {
                  ...m,
                  loading: false,
                  response: {
                    question: metadata.question || question,
                    answer: "",
                    citations: metadata.citations || [],
                    graph_context: metadata.graph_context,
                    semantic_chunks_used: metadata.semantic_chunks_used || 0,
                    graph_nodes_used: metadata.graph_nodes_used || 0,
                    model_used: metadata.model_used || "",
                    retrieval_mode: metadata.retrieval_mode || "hybrid",
                    query_type: metadata.query_type,
                    confidence_score: metadata.confidence_score,
                    timings_ms: metadata.timings_ms,
                  },
                }
                : m
            )
          );
        },
        (token) => {
          // Received streaming token chunk
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botMsgId
                ? { ...m, content: m.content + token, loading: false }
                : m
            )
          );
        },
        (error) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botMsgId
                ? { ...m, error, loading: false }
                : m
            )
          );
        },
        searchType,
        (doneData) => {
          // Received done event with confidence and timings
          setMessages((prev) =>
            prev.map((m) =>
              m.id === botMsgId
                ? {
                  ...m,
                  confidenceScore: doneData.confidence_score,
                  timingsMs: doneData.timings_ms,
                  warnings: doneData.warnings,
                }
                : m
            )
          );
        }
      );
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : "Failed to get response";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsgId
            ? { ...m, content: "", error: errMsg, loading: false }
            : m
        )
      );
    } finally {
      setThinking(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-[#252836] flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold gradient-text">GraphRAG Chat</h1>
          <p className="text-xs text-zinc-500">Semantic + Knowledge Graph + Community Insights</p>
        </div>
        <div className="flex items-center gap-3 text-sm">
          {/* Search Mode Selector */}
          <div className="flex items-center gap-1 bg-[#12141a] rounded-lg p-0.5 border border-[#252836]">
            {(Object.keys(SEARCH_TYPE_CONFIG) as SearchType[]).map((type) => {
              const cfg = SEARCH_TYPE_CONFIG[type];
              const Icon = cfg.icon;
              const isActive = searchType === type;
              return (
                <button
                  key={type}
                  onClick={() => setSearchType(type)}
                  title={cfg.desc}
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium transition-all ${isActive
                    ? `bg-[#1e2030] ${cfg.color} border border-[#353847]`
                    : "text-zinc-500 hover:text-zinc-300"
                    }`}
                >
                  <Icon size={12} />
                  {cfg.label}
                </button>
              );
            })}
          </div>

          <label className="flex items-center gap-2 text-zinc-400 cursor-pointer">
            <input
              type="checkbox"
              checked={useGraph}
              onChange={(e) => setUseGraph(e.target.checked)}
              className="accent-indigo-500"
            />
            <Share2 size={14} />
            Graph
          </label>
          <label className="flex items-center gap-2 text-zinc-400">
            <span>K:</span>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-[#1a1d27] border border-[#252836] rounded px-2 py-0.5 text-zinc-200 text-xs"
            >
              {[3, 5, 8, 10].map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
          {messages.length > 0 && (
            <button
              onClick={clearHistory}
              title="Clear chat history"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 hover:border-rose-900/50 text-zinc-400 hover:text-rose-400 text-xs transition-all"
            >
              <Trash2 size={13} />
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 flex items-center justify-center mb-4">
              <Sparkles size={28} className="text-indigo-400" />
            </div>
            <h2 className="text-lg font-semibold text-zinc-200">Ask anything about your documents</h2>
            <p className="text-zinc-500 text-sm mt-2 max-w-md">
              GRAG v2 combines vector search, knowledge graph traversal, community insights, and
              Map-Reduce global search to give you accurate, cited answers.
            </p>
            <div className="flex gap-3 mt-6">
              {(["local", "global", "hybrid"] as SearchType[]).map((type) => {
                const cfg = SEARCH_TYPE_CONFIG[type];
                const Icon = cfg.icon;
                return (
                  <div key={type} className="glass rounded-lg px-4 py-3 text-xs text-center space-y-1 min-w-[110px]">
                    <Icon size={16} className={`mx-auto ${cfg.color}`} />
                    <div className="font-medium text-zinc-300">{cfg.label} Search</div>
                    <div className="text-zinc-500">{cfg.desc}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`flex gap-3 animate-fade-in ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-indigo-600/20 flex items-center justify-center flex-shrink-0">
                <Bot size={16} className="text-indigo-400" />
              </div>
            )}

            <div className={`max-w-[80%] ${msg.role === "user" ? "order-first" : ""}`}>
              {/* Bubble */}
              <div
                className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${msg.role === "user"
                  ? "bg-indigo-600 text-white ml-auto"
                  : "glass text-zinc-100"
                  }`}
              >
                {msg.loading ? (
                  <div className="flex items-center gap-2 text-zinc-400">
                    <Loader2 size={14} className="animate-spin" />
                    <span>Thinking...</span>
                  </div>
                ) : msg.error ? (
                  <div className="flex items-center gap-2 text-rose-400">
                    <AlertCircle size={14} />
                    <span>{msg.error}</span>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>

              {/* Metadata */}
              {msg.response && (
                <div className="mt-2 space-y-2">
                  {/* Retrieval info row */}
                  <div className="flex flex-wrap gap-2 text-xs">
                    {/* Retrieval mode badge */}
                    <span className="px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                      {msg.response.retrieval_mode} retrieval
                    </span>
                    {/* Query type badge */}
                    {msg.response.query_type && (
                      <span className={`px-2 py-0.5 rounded-full border ${SEARCH_TYPE_CONFIG[msg.response.query_type as SearchType]
                        ? `${SEARCH_TYPE_CONFIG[msg.response.query_type as SearchType].color} bg-zinc-800/50 border-zinc-700/50`
                        : "text-zinc-400 bg-zinc-800/50 border-zinc-700/50"
                        }`}>
                        {msg.response.query_type}
                      </span>
                    )}
                    <span className="px-2 py-0.5 rounded-full bg-zinc-700/50 text-zinc-400">
                      {msg.response.semantic_chunks_used} chunks
                    </span>
                    {msg.response.graph_nodes_used > 0 && (
                      <span className="px-2 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400">
                        {msg.response.graph_nodes_used} graph nodes
                      </span>
                    )}
                    {/* Confidence badge */}
                    {msg.confidenceScore !== undefined && (
                      <ConfidenceBadge score={msg.confidenceScore} />
                    )}
                    {/* Total latency */}
                    {msg.timingsMs && (
                      <span className="px-2 py-0.5 rounded-full bg-zinc-800/50 border border-zinc-700/50 text-zinc-500 flex items-center gap-1">
                        <Clock size={10} />
                        {Object.values(msg.timingsMs).reduce((a, b) => a + b, 0).toFixed(0)}ms
                      </span>
                    )}
                  </div>

                  {/* Warnings */}
                  {msg.warnings && msg.warnings.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {msg.warnings.map((w, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                          ⚠ {w}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Graph context entities */}
                  {msg.response.graph_context?.entities.length ? (
                    <div className="flex flex-wrap gap-1">
                      {msg.response.graph_context.entities.map((e, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-violet-500/10 text-violet-300 border border-violet-500/20">
                          {e}
                        </span>
                      ))}
                    </div>
                  ) : null}

                  {/* Timings detail (collapsible) */}
                  {msg.timingsMs && Object.keys(msg.timingsMs).length > 1 && (
                    <details className="group">
                      <summary className="flex items-center gap-1 text-xs text-zinc-600 cursor-pointer hover:text-zinc-400 transition-colors">
                        <Clock size={10} />
                        Pipeline timings
                      </summary>
                      <div className="mt-1.5 grid grid-cols-2 gap-1 text-xs">
                        {Object.entries(msg.timingsMs)
                          .sort(([, a], [, b]) => b - a)
                          .map(([stage, ms]) => (
                            <div key={stage} className="flex justify-between px-2 py-0.5 rounded bg-zinc-800/30">
                              <span className="text-zinc-500">{stage.replace(/_/g, " ")}</span>
                              <span className="text-zinc-400 font-mono">{ms.toFixed(0)}ms</span>
                            </div>
                          ))}
                      </div>
                    </details>
                  )}

                  {/* Citations */}
                  {msg.response.citations.length > 0 && (
                    <details className="group">
                      <summary className="flex items-center gap-1 text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">
                        <BookOpen size={12} />
                        {msg.response.citations.length} citation{msg.response.citations.length > 1 ? "s" : ""}
                      </summary>
                      <div className="mt-2 space-y-1.5">
                        {msg.response.citations.map((c, i) => (
                          <CitationCard
                            key={i}
                            citation={c}
                            index={i + 1}
                            onClick={() =>
                              setActiveDrawerDoc({
                                documentId: c.document_id,
                                filename: c.document_filename,
                                chunkId: c.chunk_id,
                              })
                            }
                          />
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              )}
            </div>

            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center flex-shrink-0">
                <User size={14} className="text-zinc-300" />
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 p-4 border-t border-[#252836]">
        <div className="flex gap-3 max-w-4xl mx-auto">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 bg-[#12141a] border border-[#252836] rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50 resize-none"
            style={{ minHeight: "48px", maxHeight: "120px" }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || thinking}
            className="px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/40 rounded-xl text-white transition-all flex items-center gap-2 text-sm font-medium"
          >
            {thinking ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          </button>
        </div>
        <p className="text-center text-xs text-zinc-700 mt-2">Shift+Enter for new line · Enter to send</p>
      </div>

      {/* Slide-over Drawer for Citation & Document Inspection */}
      {activeDrawerDoc && (
        <DocumentDrawer
          documentId={activeDrawerDoc.documentId}
          documentFilename={activeDrawerDoc.filename}
          highlightChunkId={activeDrawerDoc.chunkId}
          onClose={() => setActiveDrawerDoc(null)}
        />
      )}
    </div>
  );
}

function ConfidenceBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  let color = "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  if (score < 0.4) {
    color = "text-rose-400 bg-rose-500/10 border-rose-500/20";
  } else if (score < 0.7) {
    color = "text-amber-400 bg-amber-500/10 border-amber-500/20";
  }

  return (
    <span className={`px-2 py-0.5 rounded-full border flex items-center gap-1 ${color}`}>
      <Shield size={10} />
      {pct}%
    </span>
  );
}

function CitationCard({
  citation,
  index,
  onClick,
}: {
  citation: Citation;
  index: number;
  onClick?: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className="glass rounded-lg p-2.5 text-xs space-y-1 cursor-pointer hover:border-indigo-500/50 transition-all group/card"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-zinc-300 group-hover/card:text-indigo-300 flex items-center gap-1.5">
          [{index}] {citation.document_filename}
          <ExternalLink size={10} className="opacity-0 group-hover/card:opacity-100 transition-opacity text-indigo-400" />
        </span>
        <span className="text-zinc-500">{(citation.relevance_score * 100).toFixed(0)}%</span>
      </div>
      {citation.page_number && <span className="text-zinc-600">Page {citation.page_number}</span>}
      <p className="text-zinc-500 line-clamp-2">{citation.excerpt}</p>
    </div>
  );
}
