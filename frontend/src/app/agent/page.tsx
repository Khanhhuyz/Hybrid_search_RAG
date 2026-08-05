"use client";

import { Fragment, ReactNode, useState } from "react";
import { agentApi, AgentArtifact, AgentRunResponse } from "@/lib/api";
import {
  BookOpen,
  Check,
  Download,
  FileText,
  Loader2,
  Play,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const examples = [
  "Tạo bảng so sánh các phương pháp chunking trong tài liệu, xuất Excel",
  "Viết báo cáo về kiến trúc GraphRAG và xuất PDF",
  "Lập kế hoạch triển khai GraphRAG trong 12 tuần",
  "Tạo biểu đồ các chủ đề chính được đề cập trong tài liệu",
];

export default function AgentWorkspacePage() {
  const [prompt, setPrompt] = useState("");
  const [format, setFormat] = useState("");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<AgentRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!prompt.trim() || running) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      setResult(await agentApi.run(prompt, format || undefined));
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "Không thể hoàn thành yêu cầu"
      );
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="p-5 lg:p-8 max-w-6xl mx-auto space-y-6">
      <header className="border-b border-zinc-800 pb-5">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Sparkles size={18} className="text-indigo-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-zinc-100">Agent Workspace</h1>
            <p className="text-sm text-zinc-500">
              Tạo báo cáo, kế hoạch, bảng dữ liệu và biểu đồ từ kho tri thức.
            </p>
          </div>
        </div>
      </header>

      <section className="grid lg:grid-cols-[1fr_280px] gap-4">
        <div className="bg-[#12141c] border border-zinc-800 rounded-xl p-4 space-y-3 shadow-sm">
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={5}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) run();
            }}
            placeholder="Mô tả sản phẩm hoặc công việc bạn cần Agent thực hiện..."
            className="w-full bg-zinc-950/70 border border-zinc-800 rounded-lg p-3.5 text-sm leading-6 resize-none focus:outline-none focus:border-indigo-500/70"
          />
          <div className="flex gap-3">
            <select
              value={format}
              onChange={(event) => setFormat(event.target.value)}
              className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 text-sm text-zinc-300"
            >
              <option value="">Tự chọn định dạng</option>
              <option value="md">Markdown</option>
              <option value="pdf">PDF</option>
              <option value="xlsx">Excel</option>
              <option value="csv">CSV</option>
              <option value="svg">Biểu đồ SVG</option>
            </select>
            <button
              onClick={run}
              disabled={running || !prompt.trim()}
              className="ml-auto flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
            >
              {running ? (
                <Loader2 size={15} className="animate-spin" />
              ) : (
                <Play size={15} />
              )}{" "}
              Tạo sản phẩm
            </button>
          </div>
        </div>
        <aside className="bg-[#12141c] border border-zinc-800 rounded-xl p-4">
          <p className="text-[11px] uppercase tracking-widest text-zinc-500 mb-3">
            Yêu cầu mẫu
          </p>
          <div className="space-y-2">
            {examples.map((item) => (
              <button
                key={item}
                onClick={() => setPrompt(item)}
                className="w-full text-left text-xs leading-5 p-2.5 rounded-lg border border-zinc-800 text-zinc-400 hover:text-indigo-300 hover:border-indigo-500/40 transition-colors"
              >
                {item}
              </button>
            ))}
          </div>
        </aside>
      </section>

      {error && (
        <div className="p-3.5 rounded-lg border border-rose-500/30 bg-rose-500/5 text-rose-400 text-sm">
          {error}
        </div>
      )}
      {running && (
        <div className="h-40 rounded-xl border border-zinc-800 bg-[#12141c] flex flex-col items-center justify-center gap-3 text-zinc-400">
          <Loader2 className="animate-spin text-indigo-400" />
          <div className="text-center">
            <p className="text-sm text-zinc-300">Agent đang tạo sản phẩm</p>
            <p className="text-xs text-zinc-600 mt-1">
              Lập kế hoạch · Tìm bằng chứng · Soạn nội dung · Kiểm chứng
            </p>
          </div>
        </div>
      )}

      {result && (
        <section className="grid lg:grid-cols-[260px_minmax(0,1fr)] gap-5 items-start">
          <aside className="space-y-3 lg:sticky lg:top-5">
            <div className="bg-[#12141c] border border-zinc-800 rounded-xl p-4">
              <div className="flex items-center gap-2 text-emerald-400 text-sm font-medium">
                <ShieldCheck size={16} /> Bằng chứng {statusLabel(result.evidence_status)}
              </div>
              <p className="text-xs text-zinc-500 mt-1.5">
                {result.citations.length} nguồn được sử dụng
              </p>
            </div>
            <div className="bg-[#12141c] border border-zinc-800 rounded-xl p-4">
              <p className="text-[11px] uppercase tracking-widest text-zinc-500 mb-3">
                Quá trình thực hiện
              </p>
              <ol className="space-y-3">
                {result.plan.map((step, index) => (
                  <li
                    key={`${step}-${index}`}
                    className="text-xs text-zinc-400 flex gap-2.5"
                  >
                    <span className="w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-400 flex items-center justify-center shrink-0">
                      <Check size={11} />
                    </span>
                    <span className="leading-5">{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          </aside>

          <main className="min-w-0 space-y-4">
            <article className="bg-[#12141c] border border-zinc-800 rounded-xl overflow-hidden shadow-sm">
              <header className="px-5 py-4 border-b border-zinc-800/80">
                <p className="text-[11px] font-medium text-indigo-400 uppercase tracking-widest">
                  {intentLabel(result.intent)}
                </p>
                <h2 className="text-lg font-semibold text-zinc-100 mt-1.5">
                  Kết quả từ Agent
                </h2>
                {result.answer && (
                  <p className="text-sm leading-6 text-zinc-400 mt-2">
                    {result.answer}
                  </p>
                )}
              </header>
              <div className="p-5 lg:p-7">
                {result.artifacts.length > 0 ? (
                  result.artifacts.map((artifact) => (
                    <ArtifactView key={artifact.id} artifact={artifact} />
                  ))
                ) : (
                  <RichContent content={result.answer} />
                )}
              </div>
            </article>

            {result.citations.length > 0 && (
              <details className="bg-[#12141c] border border-zinc-800 rounded-xl group">
                <summary className="px-4 py-3.5 cursor-pointer list-none flex items-center gap-2 text-sm text-zinc-300">
                  <BookOpen size={15} className="text-indigo-400" /> Nguồn tham khảo{" "}
                  <span className="text-xs text-zinc-600">
                    ({result.citations.length})
                  </span>
                </summary>
                <div className="px-4 pb-4 grid gap-2">
                  {result.citations.map((citation, index) => (
                    <div
                      key={`${citation.chunk_id}-${index}`}
                      className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3"
                    >
                      <div className="flex justify-between gap-3">
                        <p className="text-xs font-medium text-zinc-300">
                          [S{index + 1}] {citation.document_filename || "Tài liệu"}
                          {citation.page_number
                            ? ` · Trang ${citation.page_number}`
                            : ""}
                        </p>
                        <span className="text-[11px] text-emerald-500">
                          {Math.round(citation.relevance_score * 100)}%
                        </span>
                      </div>
                      <p className="text-xs leading-5 text-zinc-500 mt-1.5 line-clamp-3">
                        {citation.excerpt}
                      </p>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </main>
        </section>
      )}
    </div>
  );
}

function ArtifactView({ artifact }: { artifact: AgentArtifact }) {
  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center gap-3 rounded-lg border border-indigo-500/20 bg-indigo-500/5 p-3.5">
        <div className="w-9 h-9 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
          <FileText size={17} className="text-indigo-400" />
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium text-zinc-200 truncate">
            {artifact.title}
          </p>
          <p className="text-xs text-zinc-500 mt-0.5">
            {artifact.filename} · {formatBytes(artifact.size)}
          </p>
        </div>
        <a
          href={agentApi.downloadUrl(artifact.id)}
          className="sm:ml-auto inline-flex justify-center items-center gap-1.5 text-xs px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition-colors"
        >
          <Download size={13} /> Tải xuống
        </a>
      </div>
      <RichContent content={artifact.preview} />
    </div>
  );
}

function RichContent({ content }: { content: string }) {
  const lines = (content || "").split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) {
      index++;
      continue;
    }
    if (
      line.startsWith("| ") &&
      index + 1 < lines.length &&
      /^\|[\s|:-]+\|$/.test(lines[index + 1].trim())
    ) {
      const headers = tableCells(line);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        rows.push(tableCells(lines[index]));
        index++;
      }
      blocks.push(
        <div
          key={`table-${index}`}
          className="my-5 overflow-x-auto rounded-lg border border-zinc-700/70"
        >
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/80">
              <tr>
                {headers.map((cell, cellIndex) => (
                  <th
                    key={cellIndex}
                    className="text-left px-3.5 py-3 font-medium text-zinc-200 border-r last:border-r-0 border-zinc-700"
                  >
                    {inline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-white/[0.025]">
                  {headers.map((_, cellIndex) => (
                    <td
                      key={cellIndex}
                      className="px-3.5 py-3 align-top leading-6 text-zinc-400 border-r last:border-r-0 border-zinc-800"
                    >
                      {inline(row[cellIndex] || "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }
    if (line.startsWith("### ")) {
      blocks.push(
        <h4 key={index} className="text-sm font-semibold text-zinc-200 mt-6 mb-2">
          {inline(line.slice(4))}
        </h4>
      );
    } else if (line.startsWith("## ")) {
      blocks.push(
        <h3
          key={index}
          className="text-base font-semibold text-zinc-100 mt-7 mb-2.5 pb-2 border-b border-zinc-800"
        >
          {inline(line.slice(3))}
        </h3>
      );
    } else if (line.startsWith("# ")) {
      blocks.push(
        <h2 key={index} className="text-xl font-bold text-zinc-100 mb-3">
          {inline(line.slice(2))}
        </h2>
      );
    } else if (/^[-*] /.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*] /.test(lines[index].trim())) {
        items.push(lines[index].trim().slice(2));
        index++;
      }
      blocks.push(
        <ul
          key={`list-${index}`}
          className="my-3 ml-5 list-disc space-y-1.5 text-sm leading-6 text-zinc-400 marker:text-indigo-400"
        >
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>{inline(item)}</li>
          ))}
        </ul>
      );
      continue;
    } else {
      blocks.push(
        <p key={index} className="text-sm leading-7 text-zinc-400 my-2.5">
          {inline(line)}
        </p>
      );
    }
    index++;
  }
  return <div className="min-w-0">{blocks}</div>;
}

function inline(text: string): ReactNode {
  return text
    .split(/(\[S\d+\]|\*\*[^*]+\*\*)/g)
    .filter(Boolean)
    .map((part, index) => {
      if (/^\[S\d+\]$/.test(part)) {
        return (
          <span
            key={index}
            className="inline-flex px-1.5 py-0.5 mx-0.5 rounded bg-indigo-500/10 text-indigo-300 text-[11px] font-medium"
          >
            {part}
          </span>
        );
      }
      if (part.startsWith("**") && part.endsWith("**")) {
        return (
          <strong key={index} className="font-semibold text-zinc-200">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return <Fragment key={index}>{part.replace(/\\\|/g, "|")}</Fragment>;
    });
}

function tableCells(line: string) {
  return line
    .replace(/^\||\|$/g, "")
    .split(/(?<!\\)\|/)
    .map((cell) => cell.trim());
}

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function statusLabel(status: string) {
  return status === "sufficient"
    ? "đầy đủ"
    : status === "partial"
    ? "một phần"
    : "chưa đủ";
}

function intentLabel(intent: string) {
  const labels: Record<string, string> = {
    create_table: "Bảng dữ liệu",
    create_document: "Tài liệu",
    create_plan: "Kế hoạch",
    create_chart: "Biểu đồ",
    factual_question: "Câu trả lời",
  };
  return labels[intent] || intent.replaceAll("_", " ");
}
