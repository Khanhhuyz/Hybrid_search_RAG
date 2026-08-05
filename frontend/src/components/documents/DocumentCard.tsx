"use client";

import { useEffect, useRef, useState } from "react";
import { Document, documentsApi, ProcessingStatus } from "@/lib/api";
import {
  FileText, Trash2, Clock, CheckCircle2, AlertCircle, Loader2, Hash, GitBranch, Layers, Network, RotateCcw
} from "lucide-react";

interface DocumentCardProps {
  doc: Document;
  onDelete: (id: string) => void;
}

const statusConfig: Record<ProcessingStatus, { icon: React.ReactNode; label: string; color: string }> = {
  pending:    { icon: <Clock size={12} />,       label: "Pending",    color: "text-amber-400 bg-amber-400/5 border-amber-500/20" },
  processing: { icon: <Loader2 size={12} className="animate-spin text-indigo-400" />, label: "Processing", color: "text-indigo-400 bg-indigo-500/5 border-indigo-500/20" },
  completed:  { icon: <CheckCircle2 size={12} />, label: "Ready",    color: "text-emerald-400 bg-emerald-500/5 border-emerald-500/20" },
  failed:     { icon: <AlertCircle size={12} />,  label: "Failed",    color: "text-rose-400 bg-rose-500/5 border-rose-500/20" },
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DocumentCard({ doc: initialDoc, onDelete }: DocumentCardProps) {
  const [doc, setDoc]           = useState<Document>(initialDoc);
  const [deleting, setDeleting] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [stalled, setStalled]   = useState(false);
  const pollRef                 = useRef<NodeJS.Timeout | null>(null);

  // Poll status while processing
  useEffect(() => {
    if (doc.status === "pending" || doc.status === "processing") {
      pollRef.current = setInterval(async () => {
        try {
          const status = await documentsApi.status(doc.id);
          setDoc((current) => ({ ...current, ...status }));
          if (status.status === "completed" || status.status === "failed") {
            clearInterval(pollRef.current!);
          }
        } catch {}
      }, 3000);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [doc.id, doc.status, doc.chunk_count, doc.entity_count]);

  const handleDelete = async () => {
    if (!confirm(`Delete "${doc.original_name}"?`)) return;
    setDeleting(true);
    try {
      await documentsApi.delete(doc.id);
      onDelete(doc.id);
    } catch {
      alert("Failed to delete document");
      setDeleting(false);
    }
  };

  const handleRetry = async () => {
    setRetrying(true);
    try {
      setDoc(await documentsApi.retry(doc.id));
    } catch (error) {
      alert(error instanceof Error ? error.message : "Failed to retry document");
    } finally {
      setRetrying(false);
    }
  };

  const cfg = statusConfig[doc.status];
  const heartbeat = doc.heartbeat_at
    ? new Date(/[zZ]|[+-]\d\d:\d\d$/.test(doc.heartbeat_at) ? doc.heartbeat_at : `${doc.heartbeat_at}Z`)
    : null;
  const heartbeatTime = heartbeat?.getTime() ?? null;
  useEffect(() => {
    const check = () => setStalled(
      doc.status === "processing" && heartbeatTime !== null &&
      Date.now() - heartbeatTime > 5 * 60 * 1000
    );
    const initial = window.setTimeout(check, 0);
    const timer = window.setInterval(check, 3000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [doc.status, heartbeatTime]);
  const progress = doc.progress_total > 0
    ? Math.min(100, Math.round(doc.progress_current * 100 / doc.progress_total)) : 0;

  return (
    <div className="bg-[#12141c]/90 backdrop-blur-md rounded-lg border border-zinc-800/80 p-4 transition-all duration-200 hover:border-zinc-700/80 group">
      <div className="flex items-start justify-between gap-4">
        {/* Document Icon & Name */}
        <div className="flex items-start gap-3 min-w-0 flex-1">
          <div className="w-9 h-9 rounded bg-zinc-900 border border-zinc-800 flex items-center justify-center flex-shrink-0 text-zinc-400 group-hover:text-indigo-400 transition-colors">
            <FileText size={18} />
          </div>

          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-zinc-200 truncate tracking-tight" title={doc.original_name}>
                {doc.original_name}
              </h3>
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border ${cfg.color}`}>
                {cfg.icon}
                {cfg.label}
              </span>
            </div>

            {/* Metadata Badges */}
            <div className="flex items-center gap-3 mt-1 text-xs text-zinc-400">
              <span>{formatBytes(doc.file_size)}</span>
              <span>•</span>
              <span className="uppercase tracking-wider font-mono text-[10px] text-zinc-500">{doc.file_type.replace(".", "")}</span>
              {doc.chunk_count > 0 && (
                <>
                  <span>•</span>
                  <span className="flex items-center gap-1 text-zinc-400">
                    <Hash size={11} className="text-zinc-500" />
                    {doc.chunk_count} chunks
                  </span>
                </>
              )}
              {doc.entity_count > 0 && (
                <>
                  <span>•</span>
                  <span className="flex items-center gap-1 text-indigo-300">
                    <GitBranch size={11} className="text-indigo-400" />
                    {doc.entity_count} entities
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1">
        {(doc.status === "failed" || stalled || !!doc.error_message) && <button
          onClick={handleRetry}
          disabled={retrying}
          className="p-1.5 rounded border border-amber-500/20 text-amber-400 hover:bg-amber-500/10"
          title="Retry processing"
        >{retrying ? <Loader2 size={15} className="animate-spin" /> : <RotateCcw size={15} />}</button>}
        <button
          onClick={handleDelete}
          disabled={deleting}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-rose-500/10 rounded border border-transparent hover:border-rose-500/20 text-zinc-500 hover:text-rose-400"
          title="Delete document"
        >
          {deleting ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
        </button>
        </div>
      </div>

      {/* 4-Step Pipeline Stepper for Processing Documents */}
      {doc.status === "processing" && (
        <div className="mt-3.5 pt-3 border-t border-zinc-800/60 space-y-2">
        <div className="flex justify-between text-[11px] text-zinc-400">
          <span className="capitalize">{stalled ? "Stalled" : doc.progress_stage}</span>
          <span>{doc.progress_total > 0 ? `${doc.progress_current}/${doc.progress_total} (${progress}%)` : "Starting..."}</span>
        </div>
        <div className="h-1.5 rounded bg-zinc-800 overflow-hidden"><div className="h-full bg-indigo-500 transition-all" style={{ width: `${progress}%` }} /></div>
        <div className="grid grid-cols-4 gap-2 text-xs">
          <StepBadge step={1} label="Read" active={true} done={true} icon={<FileText size={10} />} />
          <StepBadge step={2} label="Chunk" active={doc.progress_stage === "extracting"} done={["embedding", "graph", "completed"].includes(doc.progress_stage)} icon={<Layers size={10} />} />
          <StepBadge step={3} label="Embed" active={doc.progress_stage === "embedding"} done={["graph", "completed"].includes(doc.progress_stage)} icon={<Hash size={10} />} />
          <StepBadge step={4} label="Graph" active={doc.progress_stage === "graph"} done={doc.progress_stage === "completed"} icon={<Network size={10} />} />
        </div>
        </div>
      )}

      {/* Error Output */}
      {doc.error_message && (
        <div className="mt-2.5 p-2 rounded bg-rose-500/5 border border-rose-500/20 text-xs text-rose-400">
          <p className="font-mono">{doc.error_message}</p>
        </div>
      )}
    </div>
  );
}

function StepBadge({ label, active, done, icon }: { step: number; label: string; active: boolean; done: boolean; icon: React.ReactNode }) {
  return (
    <div className={`flex items-center gap-1.5 p-1.5 rounded border text-[11px] ${
      done
        ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
        : active
        ? "border-indigo-500/30 bg-indigo-500/10 text-indigo-300 animate-pulse"
        : "border-zinc-800 bg-zinc-900/50 text-zinc-600"
    }`}>
      {done ? <CheckCircle2 size={10} className="text-emerald-400" /> : icon}
      <span className="font-medium tracking-tight truncate">{label}</span>
    </div>
  );
}
