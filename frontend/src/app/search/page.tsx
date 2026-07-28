"use client";

import { useState } from "react";
import { searchApi, SearchResponse } from "@/lib/api";
import { Search, Loader2, FileText, Hash, BarChart2, Share2 } from "lucide-react";

type SearchMode = "semantic" | "graph";

export default function SearchPage() {
  const [query, setQuery]         = useState("");
  const [mode, setMode]           = useState<SearchMode>("semantic");
  const [topK, setTopK]           = useState(5);
  const [loading, setLoading]     = useState(false);
  const [results, setResults]     = useState<SearchResponse | null>(null);
  const [error, setError]         = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = mode === "semantic"
        ? await searchApi.semantic(query.trim(), topK)
        : await searchApi.graph(query.trim(), topK);
      setResults(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold gradient-text">Search</h1>
        <p className="text-sm text-zinc-500 mt-1">Semantic vector search or knowledge graph lookup</p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="space-y-4">
        {/* Mode Switch */}
        <div className="flex gap-2">
          {(["semantic", "graph"] as SearchMode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === m
                  ? "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30"
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-white/5"
              }`}
            >
              {m === "semantic" ? <BarChart2 size={14} /> : <Share2 size={14} />}
              {m === "semantic" ? "Semantic Search" : "Graph Search"}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="flex gap-3">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                mode === "semantic"
                  ? "Search for content in your documents..."
                  : "Search for an entity (person, org, concept)..."
              }
              className="w-full pl-10 pr-4 py-3 bg-[#12141a] border border-[#252836] rounded-xl text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50"
            />
          </div>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="px-3 py-3 bg-[#12141a] border border-[#252836] rounded-xl text-zinc-200 text-sm"
          >
            {[3, 5, 8, 10].map((k) => <option key={k} value={k}>Top {k}</option>)}
          </select>
          <button
            type="submit"
            disabled={!query.trim() || loading}
            className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/40 rounded-xl text-white text-sm font-medium transition-all flex items-center gap-2"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
            Search
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-zinc-300">
              {results.total} results for &quot;<span className="text-indigo-400">{results.query}</span>&quot;
            </h2>
            <span className="text-xs px-2 py-1 rounded-full bg-zinc-800 text-zinc-500">
              {results.search_type} search
            </span>
          </div>

          <div className="space-y-3">
            {results.results.map((result, i) => (
              <div key={result.chunk.id} className="glass rounded-xl p-4 space-y-2 hover:border-indigo-500/20 transition-all animate-fade-in">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-2 text-xs text-zinc-500">
                    <FileText size={12} />
                    <span className="text-zinc-400 font-medium">{result.document_filename}</span>
                    {result.chunk.page_number && (
                      <span>· Page {result.chunk.page_number}</span>
                    )}
                    {result.chunk.section && (
                      <span className="text-zinc-600">· {result.chunk.section}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-xs text-emerald-400 font-medium flex-shrink-0">
                    <BarChart2 size={10} />
                    {(result.score * 100).toFixed(1)}%
                  </div>
                </div>

                <p className="text-sm text-zinc-300 leading-relaxed line-clamp-4">
                  {result.chunk.content}
                </p>

                <div className="flex items-center gap-2 text-xs text-zinc-600">
                  <Hash size={10} />
                  <span>Chunk {result.chunk.chunk_index}</span>
                </div>
              </div>
            ))}
          </div>

          {results.total === 0 && (
            <div className="text-center py-12 text-zinc-500">
              <Search size={32} className="mx-auto mb-3 text-zinc-700" />
              <p>No results found. Try a different query.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
