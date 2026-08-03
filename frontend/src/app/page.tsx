"use client";

import { useCallback, useEffect, useState, useMemo } from "react";
import { Document, documentsApi, healthApi, HealthStatus, graphApi, GraphStats } from "@/lib/api";
import { UploadZone } from "@/components/documents/UploadZone";
import { DocumentCard } from "@/components/documents/DocumentCard";
import { FileText, Share2, Cpu, MessageSquare, RefreshCw, Search, Filter } from "lucide-react";

export default function DashboardPage() {
  const [documents, setDocuments]   = useState<Document[]>([]);
  const [health, setHealth]         = useState<HealthStatus | null>(null);
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [loading, setLoading]       = useState(true);

  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [typeFilter, setTypeFilter] = useState<string>("all");

  const loadData = useCallback(async () => {
    try {
      const [docsRes, healthRes, statsRes] = await Promise.allSettled([
        documentsApi.list(),
        healthApi.check(),
        graphApi.stats(),
      ]);
      if (docsRes.status === "fulfilled") setDocuments(docsRes.value.documents);
      if (healthRes.status === "fulfilled") setHealth(healthRes.value);
      if (statsRes.status === "fulfilled") setGraphStats(statsRes.value);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const handleUploaded = (doc: Document) => {
    setDocuments((prev) => [doc, ...prev]);
  };

  const handleDelete = (id: string) => {
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  };

  const completedDocs = documents.filter((d) => d.status === "completed");
  const totalChunks   = completedDocs.reduce((s, d) => s + d.chunk_count, 0);

  // Filtered documents
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      const matchesSearch = doc.original_name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === "all" || doc.status === statusFilter;
      const matchesType   = typeFilter === "all" || doc.file_type.toLowerCase().includes(typeFilter.toLowerCase());
      return matchesSearch && matchesStatus && matchesType;
    });
  }, [documents, searchQuery, statusFilter, typeFilter]);

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 pb-5">
        <div>
          <h1 className="text-xl font-bold text-zinc-100 tracking-tight flex items-center gap-2">
            Dashboard
            <span className="text-xs font-mono font-normal text-zinc-500 px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800">v1.0.0</span>
          </h1>
          <p className="text-xs text-zinc-400 mt-1">Ingest, process, and inspect your knowledge graph assets</p>
        </div>
        <button
          onClick={loadData}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-zinc-400 hover:text-zinc-200 text-xs font-medium transition-all"
        >
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          Sync
        </button>
      </div>

      {/* Stats Cards - Minimalist Editorial */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3.5">
        <StatCard
          icon={<FileText size={15} className="text-indigo-400" />}
          label="Total Documents"
          value={String(documents.length)}
          sub={`${completedDocs.length} ready • ${documents.length - completedDocs.length} active`}
        />
        <StatCard
          icon={<Share2 size={15} className="text-violet-400" />}
          label="Knowledge Graph"
          value={String(graphStats?.nodes ?? 0)}
          sub={`${graphStats?.edges ?? 0} total relationships`}
        />
        <StatCard
          icon={<MessageSquare size={15} className="text-emerald-400" />}
          label="Vector Chunks"
          value={String(totalChunks)}
          sub="Indexed in local Qdrant"
        />
        <StatCard
          icon={<Cpu size={15} className={(typeof health?.services?.ollama === "object" ? health.services.ollama.status : health?.services?.ollama) === "ok" ? "text-emerald-400" : "text-rose-400"} />}
          label="AI Inference Engine"
          value={(typeof health?.services?.ollama === "object" ? health.services.ollama.status : health?.services?.ollama) === "ok" ? "Online" : "Offline"}
          sub={(typeof health?.services?.ollama === "object" ? health.services.ollama.model : undefined) ?? "Ollama Local"}
        />
      </div>

      {/* Upload Zone Section */}
      <section className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest font-mono">Document Ingestion</h2>
          <span className="text-[11px] text-zinc-500">Max size 500 MB</span>
        </div>
        <UploadZone onUploaded={handleUploaded} />
      </section>

      {/* Document List & Search Filter */}
      <section className="space-y-3.5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-zinc-800/80 pb-3">
          <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-widest font-mono">
            Repository ({filteredDocuments.length}/{documents.length})
          </h2>

          {/* Search and Filters */}
          <div className="flex items-center gap-2 w-full sm:w-auto flex-wrap">
            {/* Search Input */}
            <div className="relative flex-1 sm:w-48">
              <Search size={13} className="absolute left-2.5 top-2.5 text-zinc-500" />
              <input
                type="text"
                placeholder="Filter files..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-zinc-900 border border-zinc-800 rounded pl-8 pr-3 py-1 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-zinc-700"
              />
            </div>

            {/* Status Selector */}
            <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-800 rounded p-0.5 text-xs">
              <Filter size={11} className="text-zinc-500 ml-1.5" />
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="bg-transparent text-zinc-400 text-[11px] focus:outline-none pr-1 cursor-pointer"
              >
                <option value="all" className="bg-zinc-900 text-zinc-300">All Status</option>
                <option value="completed" className="bg-zinc-900 text-zinc-300">Ready</option>
                <option value="processing" className="bg-zinc-900 text-zinc-300">Processing</option>
                <option value="failed" className="bg-zinc-900 text-zinc-300">Failed</option>
              </select>
            </div>

            {/* Extension Selector */}
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="bg-zinc-900 border border-zinc-800 rounded text-zinc-400 text-[11px] px-2 py-1 focus:outline-none cursor-pointer"
            >
              <option value="all" className="bg-zinc-900 text-zinc-300">All Formats</option>
              <option value="pdf" className="bg-zinc-900 text-zinc-300">PDF</option>
              <option value="docx" className="bg-zinc-900 text-zinc-300">DOCX</option>
              <option value="txt" className="bg-zinc-900 text-zinc-300">TXT</option>
              <option value="md" className="bg-zinc-900 text-zinc-300">MD</option>
            </select>
          </div>
        </div>

        {/* List Content */}
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 rounded bg-zinc-900/50 border border-zinc-800/50 animate-pulse" />
            ))}
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-14 text-center rounded border border-dashed border-zinc-800/80 bg-zinc-900/20">
            <FileText size={32} className="text-zinc-600 mb-2" />
            <p className="text-xs font-medium text-zinc-400">No matching documents found</p>
            <p className="text-xs text-zinc-600 mt-0.5">Try clearing your filters or upload a new file above</p>
          </div>
        ) : (
          <div className="space-y-2">
            {filteredDocuments.map((doc) => (
              <DocumentCard key={doc.id} doc={doc} onDelete={handleDelete} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function StatCard({
  icon, label, value, sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="bg-[#12141c]/90 rounded border border-zinc-800 p-3.5 transition-colors hover:border-zinc-700">
      <div className="flex items-center gap-1.5 mb-1.5">
        {icon}
        <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-lg font-bold text-zinc-100 tracking-tight">{value}</p>
      <p className="text-[11px] text-zinc-500 mt-0.5 truncate">{sub}</p>
    </div>
  );
}
