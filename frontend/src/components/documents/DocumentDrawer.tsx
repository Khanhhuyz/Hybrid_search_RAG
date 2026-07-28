"use client";

import { useEffect, useState, useRef } from "react";
import { documentsApi, ChunkResult } from "@/lib/api";
import { X, FileText, BookOpen, Loader2, ExternalLink, Eye } from "lucide-react";

interface DocumentDrawerProps {
  documentId: string | null;
  documentFilename?: string;
  highlightChunkId?: string;
  onClose: () => void;
}

export function DocumentDrawer({
  documentId,
  documentFilename,
  highlightChunkId,
  onClose,
}: DocumentDrawerProps) {
  const [chunks, setChunks] = useState<ChunkResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [filename, setFilename] = useState(documentFilename || "");
  const [focusOnly, setFocusOnly] = useState(Boolean(highlightChunkId));

  const drawerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!documentId) return;

    setLoading(true);
    documentsApi
      .getChunks(documentId)
      .then((res) => {
        setChunks(res.chunks || []);
        if (res.filename) setFilename(res.filename);
      })
      .catch((err) => {
        console.error("Failed to load document chunks:", err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [documentId]);

  // Auto-scroll to highlighted chunk when chunks finish loading
  useEffect(() => {
    if (!loading && highlightChunkId && chunks.length > 0) {
      setTimeout(() => {
        const el = document.getElementById(`chunk-${highlightChunkId}`);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 150);
    }
  }, [loading, highlightChunkId, chunks, focusOnly]);

  if (!documentId) return null;

  // Filter chunks if focus mode is ON
  const visibleChunks = focusOnly && highlightChunkId
    ? chunks.filter((c) => c.id === highlightChunkId)
    : chunks;

  const rawFileUrl = `http://localhost:8000/api/v1/documents/${documentId}/file`;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm animate-fade-in">
      {/* Backdrop click to close */}
      <div className="flex-1" onClick={onClose} />

      {/* Slide-over Panel */}
      <div
        ref={drawerRef}
        className="w-full max-w-xl bg-[#12141c] border-l border-zinc-800 h-full flex flex-col shadow-2xl animate-slide-left"
      >
        {/* Header */}
        <div className="p-4 border-b border-zinc-800 flex items-center justify-between bg-zinc-900/50">
          <div className="flex items-center gap-2 overflow-hidden">
            <FileText size={18} className="text-indigo-400 flex-shrink-0" />
            <div className="truncate">
              <h2 className="text-sm font-bold text-zinc-100 truncate">{filename || "Document Viewer"}</h2>
              <p className="text-[11px] text-zinc-500 font-mono">ID: {documentId.substring(0, 8)}...</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={rawFileUrl}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1 text-xs px-2.5 py-1.5 rounded-lg bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/30 transition-all font-medium"
              title="Open raw PDF file in new tab"
            >
              <ExternalLink size={12} />
              Open File
            </a>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-all"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* View Mode Filter Bar */}
        {highlightChunkId && (
          <div className="px-4 py-2 bg-zinc-950/60 border-b border-zinc-800/60 flex items-center justify-between text-xs">
            <span className="text-zinc-400 flex items-center gap-1.5">
              <Eye size={12} className="text-indigo-400" />
              Showing {visibleChunks.length} of {chunks.length} chunks
            </span>
            <div className="flex items-center gap-1 bg-zinc-900 p-0.5 rounded-lg border border-zinc-800">
              <button
                onClick={() => setFocusOnly(true)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                  focusOnly
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                Focus Target Chunk
              </button>
              <button
                onClick={() => setFocusOnly(false)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                  !focusOnly
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                All Chunks
              </button>
            </div>
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-64 text-zinc-500 space-y-2">
              <Loader2 size={24} className="animate-spin text-indigo-400" />
              <p className="text-xs">Loading document chunks...</p>
            </div>
          ) : visibleChunks.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-zinc-500 space-y-2">
              <BookOpen size={28} className="text-zinc-600" />
              <p className="text-xs">No chunks found for this view</p>
            </div>
          ) : (
            visibleChunks.map((c, i) => {
              const isHighlighted = highlightChunkId === c.id;
              return (
                <div
                  key={c.id || i}
                  id={`chunk-${c.id}`}
                  className={`p-4 rounded-xl border text-xs leading-relaxed transition-all ${
                    isHighlighted
                      ? "bg-indigo-950/50 border-indigo-500 shadow-xl ring-2 ring-indigo-500/50"
                      : "bg-zinc-900/40 border-zinc-800/80 hover:border-zinc-700"
                  }`}
                >
                  <div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-800/60 text-zinc-400">
                    <span className="font-mono text-[10px] text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20 font-bold">
                      Chunk #{c.chunk_index + 1} {isHighlighted && "★ Target Chunk"}
                    </span>
                    {c.page_number && (
                      <span className="text-[10px] text-zinc-400 font-medium">Page {c.page_number}</span>
                    )}
                  </div>
                  {c.section && (
                    <p className="text-[11px] font-semibold text-zinc-300 mb-1.5 font-mono">
                      {c.section}
                    </p>
                  )}
                  <p className="text-zinc-200 whitespace-pre-wrap text-sm leading-relaxed">{c.content}</p>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
