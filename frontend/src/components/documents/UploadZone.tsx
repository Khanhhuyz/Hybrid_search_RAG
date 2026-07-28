"use client";

import { useCallback, useState } from "react";
import { Upload, FileText, AlertCircle, CheckCircle2 } from "lucide-react";
import { documentsApi, Document } from "@/lib/api";

interface UploadZoneProps {
  onUploaded: (doc: Document) => void;
}

export function UploadZone({ onUploaded }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading]   = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [success, setSuccess]       = useState<string | null>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return;
      setError(null);
      setSuccess(null);

      const allowed = [".pdf", ".docx", ".txt", ".md"];
      const file = files[0];
      const ext  = "." + file.name.split(".").pop()?.toLowerCase();

      if (!allowed.includes(ext)) {
        setError(`Unsupported file type: ${ext}. Allowed: ${allowed.join(", ")}`);
        return;
      }

      if (file.size > 500 * 1024 * 1024) {
        setError("File too large. Maximum size: 500 MB");
        return;
      }

      setUploading(true);
      try {
        const doc = await documentsApi.upload(file);
        setSuccess(`"${file.name}" uploaded! Processing in background...`);
        onUploaded(doc);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [onUploaded]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  return (
    <div className="space-y-3">
      <label
        className={`relative flex flex-col items-center justify-center gap-3 p-8 rounded-xl border-2 border-dashed cursor-pointer transition-all duration-200 ${
          isDragging
            ? "border-indigo-400 bg-indigo-500/10"
            : "border-[#252836] hover:border-indigo-500/50 hover:bg-indigo-500/5 bg-[#12141a]"
        } ${uploading ? "pointer-events-none opacity-60" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file"
          className="sr-only"
          accept=".pdf,.docx,.txt,.md"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={uploading}
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-2">
            <div className="w-10 h-10 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-zinc-400">Uploading...</p>
          </div>
        ) : (
          <>
            <div className="w-12 h-12 rounded-xl bg-indigo-600/20 flex items-center justify-center">
              <Upload size={22} className="text-indigo-400" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-zinc-200">
                Drop a file here, or <span className="text-indigo-400">click to browse</span>
              </p>
              <p className="text-xs text-zinc-500 mt-1">
                PDF, DOCX, TXT, Markdown · Max 500 MB
              </p>
            </div>
          </>
        )}
      </label>

      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-sm animate-fade-in">
          <AlertCircle size={16} className="flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {success && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm animate-fade-in">
          <CheckCircle2 size={16} className="flex-shrink-0 mt-0.5" />
          <span>{success}</span>
        </div>
      )}
    </div>
  );
}
