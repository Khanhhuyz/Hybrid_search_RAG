"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { graphApi, GraphData, GraphStats } from "@/lib/api";
import { Search, RefreshCw, Loader2, Share2 } from "lucide-react";

// D3 is browser-only — disable SSR
const GraphVisualizer = dynamic(
  () => import("@/components/graph/GraphVisualizer").then((m) => m.GraphVisualizer),
  { ssr: false }
);

export default function GraphPage() {
  const [graphData, setGraphData]   = useState<GraphData | null>(null);
  const [stats, setStats]           = useState<GraphStats | null>(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching]     = useState(false);

  const loadFullGraph = async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, graphStats] = await Promise.all([
        graphApi.visualize(undefined, 200),
        graphApi.stats(),
      ]);
      setGraphData(data);
      setStats(graphStats);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load graph");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const timer = window.setTimeout(() => void loadFullGraph(), 0);
    return () => window.clearTimeout(timer);
  }, []);

  const handleEntitySearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) { loadFullGraph(); return; }
    setSearching(true);
    setError(null);
    try {
      const data = await graphApi.queryEntity(searchQuery.trim(), 2);
      setGraphData(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Entity not found");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-[#252836]">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-lg font-bold gradient-text">Knowledge Graph</h1>
            <p className="text-xs text-zinc-500">Entity relationships extracted from your documents</p>
          </div>

          {/* Stats */}
          {stats && (
            <div className="flex gap-4 text-sm">
              <div className="glass px-3 py-1.5 rounded-lg text-center">
                <p className="text-lg font-bold text-violet-400">{stats.nodes}</p>
                <p className="text-xs text-zinc-500">Nodes</p>
              </div>
              <div className="glass px-3 py-1.5 rounded-lg text-center">
                <p className="text-lg font-bold text-indigo-400">{stats.edges}</p>
                <p className="text-xs text-zinc-500">Edges</p>
              </div>
            </div>
          )}

          {/* Controls */}
          <div className="flex items-center gap-2">
            <form onSubmit={handleEntitySearch} className="flex gap-2">
              <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search entity..."
                  className="pl-8 pr-3 py-2 bg-[#12141a] border border-[#252836] rounded-lg text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-500/50 w-44"
                />
              </div>
              <button
                type="submit"
                disabled={searching}
                className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white text-sm transition-all"
              >
                {searching ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
              </button>
            </form>
            <button
              onClick={loadFullGraph}
              className="p-2 hover:bg-white/5 rounded-lg text-zinc-500 hover:text-zinc-300 transition-all"
              title="Reset to full graph"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Graph Canvas */}
      <div className="flex-1 relative overflow-hidden bg-[#0a0b0f]">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-2xl bg-violet-600/20 flex items-center justify-center">
                <Share2 size={24} className="text-violet-400 animate-pulse" />
              </div>
              <p className="text-zinc-500 text-sm">Loading knowledge graph...</p>
            </div>
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="glass rounded-xl p-6 text-center max-w-sm">
              <p className="text-rose-400 font-medium mb-2">Failed to load graph</p>
              <p className="text-zinc-500 text-sm mb-4">{error}</p>
              <button onClick={loadFullGraph} className="px-4 py-2 bg-indigo-600 rounded-lg text-white text-sm">
                Retry
              </button>
            </div>
          </div>
        ) : graphData && graphData.nodes.length === 0 ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <Share2 size={48} className="text-zinc-700 mx-auto mb-3" />
              <p className="text-zinc-400 font-medium">No graph data yet</p>
              <p className="text-zinc-600 text-sm mt-1">Upload and process documents to build the knowledge graph</p>
            </div>
          </div>
        ) : graphData ? (
          <GraphVisualizer data={graphData} />
        ) : null}
      </div>

      {/* Entity Type Stats */}
      {stats?.entity_types && Object.keys(stats.entity_types).length > 0 && (
        <div className="flex-shrink-0 px-6 py-3 border-t border-[#252836] flex items-center gap-4 overflow-x-auto">
          <span className="text-xs text-zinc-600 flex-shrink-0">Entity types:</span>
          {Object.entries(stats.entity_types).map(([type, count]) => (
            <span key={type} className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 flex-shrink-0">
              {type} ({count})
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
