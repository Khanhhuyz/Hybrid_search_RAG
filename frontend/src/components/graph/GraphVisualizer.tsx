"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { GraphData, GraphNode } from "@/lib/api";
import { Search } from "lucide-react";

// Entity type → color mapping
const TYPE_COLORS: Record<string, string> = {
  PERSON:       "#6366f1",
  ORGANIZATION: "#8b5cf6",
  COURSE:       "#10b981",
  DEPARTMENT:   "#f59e0b",
  PRODUCT:      "#3b82f6",
  LOCATION:     "#f43f5e",
  CONCEPT:      "#64748b",
};

interface GraphVisualizerProps {
  data: GraphData;
  width?: number;
  height?: number;
  highlightEntity?: string;
}

interface D3Node extends d3.SimulationNodeDatum, GraphNode {
  x?: number;
  y?: number;
}

interface D3Link extends d3.SimulationLinkDatum<D3Node> {
  source: D3Node;
  target: D3Node;
  relation: string;
  weight: number;
}

export function GraphVisualizer({ data, width = 800, height = 600, highlightEntity }: GraphVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selected, setSelected] = useState<D3Node | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter]   = useState("ALL");

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Filter nodes based on user search & type selection
    const filteredNodesData = data.nodes.filter((n) => {
      const matchesSearch = !searchQuery || n.label.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesType   = typeFilter === "ALL" || n.type === typeFilter;
      return matchesSearch && matchesType;
    });

    const validNodeIds = new Set(filteredNodesData.map((n) => n.id));

    const nodeMap = new Map<string, D3Node>();
    const nodes: D3Node[] = filteredNodesData.map((n) => {
      const d: D3Node = { ...n };
      nodeMap.set(n.id, d);
      return d;
    });

    const links: D3Link[] = data.edges
      .filter((e) => validNodeIds.has(e.source) && validNodeIds.has(e.target))
      .map((e) => {
        const src = nodeMap.get(e.source);
        const tgt = nodeMap.get(e.target);
        if (!src || !tgt) return null;
        return { source: src, target: tgt, relation: e.relation, weight: e.weight } as D3Link;
      })
      .filter(Boolean) as D3Link[];

    // ── Simulation ─────────────────────────────────────────────────────────
    const simulation = d3
      .forceSimulation<D3Node>(nodes)
      .force("link", d3.forceLink<D3Node, D3Link>(links).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(32));

    // ── Container with zoom ────────────────────────────────────────────────
    const zoom = d3.zoom<SVGSVGElement, unknown>().scaleExtent([0.2, 4]).on("zoom", (event) => {
      g.attr("transform", event.transform);
    });
    svg.call(zoom);

    const g = svg.append("g");

    // ── Arrow marker ───────────────────────────────────────────────────────
    svg.append("defs").append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#374151");

    // ── Links ──────────────────────────────────────────────────────────────
    const link = g.append("g").selectAll("line").data(links).enter().append("line")
      .attr("stroke", "#374151")
      .attr("stroke-width", (d) => Math.min(d.weight, 3))
      .attr("stroke-opacity", 0.6)
      .attr("marker-end", "url(#arrow)");

    // ── Link labels ────────────────────────────────────────────────────────
    const linkLabel = g.append("g").selectAll("text").data(links).enter().append("text")
      .attr("font-size", "8px")
      .attr("fill", "#6b7280")
      .attr("text-anchor", "middle")
      .text((d) => d.relation.replace(/_/g, " ").toLowerCase());

    // ── Nodes ──────────────────────────────────────────────────────────────
    const node = g.append("g")
      .selectAll<SVGGElement, D3Node>("g")
      .data(nodes)
      .enter()
      .append("g")
      .attr("cursor", "pointer")
      .call(
        d3.drag<SVGGElement, D3Node>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      )
      .on("click", (_, d) => setSelected((prev) => prev?.id === d.id ? null : d));

    node.append("circle")
      .attr("r", (d) => (highlightEntity && d.label.toLowerCase().includes(highlightEntity.toLowerCase()) ? 18 : 14))
      .attr("fill", (d) => TYPE_COLORS[d.type] || "#64748b")
      .attr("fill-opacity", 0.9)
      .attr("stroke", (d) => (highlightEntity && d.label.toLowerCase().includes(highlightEntity.toLowerCase()) ? "#ffffff" : TYPE_COLORS[d.type] || "#64748b"))
      .attr("stroke-width", (d) => (highlightEntity && d.label.toLowerCase().includes(highlightEntity.toLowerCase()) ? 3 : 2))
      .attr("stroke-opacity", 0.8);

    node.append("text")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("font-weight", "600")
      .attr("fill", "white")
      .text((d) => d.label.substring(0, 2).toUpperCase());

    node.append("title").text((d) => `${d.label} [${d.type}]`);

    node.append("text")
      .attr("dy", "2.2em")
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("fill", "#94a3b8")
      .text((d) => d.label.length > 14 ? d.label.substring(0, 12) + "…" : d.label);

    // ── Tick ──────────────────────────────────────────────────────────────
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x!)
        .attr("y1", (d) => d.source.y!)
        .attr("x2", (d) => d.target.x!)
        .attr("y2", (d) => d.target.y!);
      linkLabel
        .attr("x", (d) => ((d.source.x ?? 0) + (d.target.x ?? 0)) / 2)
        .attr("y", (d) => ((d.source.y ?? 0) + (d.target.y ?? 0)) / 2);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    return () => { simulation.stop(); };
  }, [data, width, height, searchQuery, typeFilter, highlightEntity]);

  return (
    <div className="relative w-full h-full flex flex-col">
      {/* Search & Filter bar */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-[#12141c]/90 border border-zinc-800 rounded-lg p-1.5 backdrop-blur-md">
        <div className="relative flex items-center">
          <Search size={13} className="absolute left-2.5 text-zinc-500" />
          <input
            type="text"
            placeholder="Search entity..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-zinc-900 border border-zinc-800 text-xs text-zinc-200 rounded pl-7 pr-2.5 py-1 w-36 focus:outline-none focus:border-zinc-700"
          />
        </div>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
          className="bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 rounded px-2 py-1 focus:outline-none cursor-pointer"
        >
          <option value="ALL">All Types</option>
          {Object.keys(TYPE_COLORS).map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>

      <svg ref={svgRef} width={width} height={height} className="w-full h-full" />

      {/* Legend */}
      <div className="absolute top-3 right-3 glass rounded-lg p-3 space-y-1.5 text-xs">
        {Object.entries(TYPE_COLORS).map(([type, color]) => (
          <div key={type} className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-zinc-400 capitalize">{type.toLowerCase()}</span>
          </div>
        ))}
      </div>

      {/* Selected node info */}
      {selected && (
        <div className="absolute bottom-3 left-3 glass rounded-lg p-3 text-xs max-w-[220px] animate-fade-in space-y-1">
          <p className="font-semibold text-zinc-100">{selected.label}</p>
          <span className="inline-block px-2 py-0.5 rounded text-[10px] bg-indigo-500/20 text-indigo-300 font-mono">
            {selected.type}
          </span>
          <p className="text-zinc-500 text-[11px] mt-1">{selected.document_ids?.length ?? 0} source document(s)</p>
        </div>
      )}
    </div>
  );
}

