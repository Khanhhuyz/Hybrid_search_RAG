"use client";

import { useState, useRef, useEffect } from "react";
import { chatApi, ChatMessage, ChatResponse, Citation } from "@/lib/api";
import {
  Send, Bot, User, BookOpen, Share2, Loader2, Sparkles, AlertCircle
} from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
  error?: string;
  loading?: boolean;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState("");
  const [useGraph, setUseGraph] = useState(true);
  const [topK, setTopK]         = useState(5);
  const [thinking, setThinking] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    const question = input.trim();
    if (!question || thinking) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: question };
    const botMsg:  Message = { id: (Date.now() + 1).toString(), role: "assistant", content: "", loading: true };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput("");
    setThinking(true);

    // Build history for context
    const history: ChatMessage[] = messages
      .filter((m) => !m.loading && !m.error)
      .slice(-10)
      .map((m) => ({ role: m.role, content: m.content }));

    try {
      const resp = await chatApi.ask(question, history, useGraph, topK);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsg.id
            ? { ...m, content: resp.answer, response: resp, loading: false }
            : m
        )
      );
    } catch (e: unknown) {
      const errMsg = e instanceof Error ? e.message : "Failed to get response";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botMsg.id
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
          <p className="text-xs text-zinc-500">Hybrid semantic + knowledge graph retrieval</p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <label className="flex items-center gap-2 text-zinc-400 cursor-pointer">
            <input
              type="checkbox"
              checked={useGraph}
              onChange={(e) => setUseGraph(e.target.checked)}
              className="accent-indigo-500"
            />
            <Share2 size={14} />
            Use Graph
          </label>
          <label className="flex items-center gap-2 text-zinc-400">
            <span>Top-K:</span>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-[#1a1d27] border border-[#252836] rounded px-2 py-0.5 text-zinc-200 text-xs"
            >
              {[3, 5, 8, 10].map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
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
            <p className="text-zinc-500 text-sm mt-2 max-w-sm">
              GRAG combines vector search and knowledge graph traversal to give you accurate, cited answers.
            </p>
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
                className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
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
                  {/* Retrieval info */}
                  <div className="flex flex-wrap gap-2 text-xs">
                    <span className="px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                      {msg.response.retrieval_mode} retrieval
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-zinc-700/50 text-zinc-400">
                      {msg.response.semantic_chunks_used} chunks
                    </span>
                    {msg.response.graph_nodes_used > 0 && (
                      <span className="px-2 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400">
                        {msg.response.graph_nodes_used} graph nodes
                      </span>
                    )}
                  </div>

                  {/* Graph context */}
                  {msg.response.graph_context?.entities.length ? (
                    <div className="flex flex-wrap gap-1">
                      {msg.response.graph_context.entities.map((e, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-violet-500/10 text-violet-300 border border-violet-500/20">
                          {e}
                        </span>
                      ))}
                    </div>
                  ) : null}

                  {/* Citations */}
                  {msg.response.citations.length > 0 && (
                    <details className="group">
                      <summary className="flex items-center gap-1 text-xs text-zinc-500 cursor-pointer hover:text-zinc-300 transition-colors">
                        <BookOpen size={12} />
                        {msg.response.citations.length} citation{msg.response.citations.length > 1 ? "s" : ""}
                      </summary>
                      <div className="mt-2 space-y-1.5">
                        {msg.response.citations.map((c, i) => (
                          <CitationCard key={i} citation={c} index={i + 1} />
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
    </div>
  );
}

function CitationCard({ citation, index }: { citation: Citation; index: number }) {
  return (
    <div className="glass rounded-lg p-2.5 text-xs space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-zinc-300">[{index}] {citation.document_filename}</span>
        <span className="text-zinc-500">{(citation.relevance_score * 100).toFixed(0)}%</span>
      </div>
      {citation.page_number && <span className="text-zinc-600">Page {citation.page_number}</span>}
      <p className="text-zinc-500 line-clamp-2">{citation.excerpt}</p>
    </div>
  );
}
