import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/layout/Sidebar";

export const metadata: Metadata = {
  title: "GRAG — GraphRAG System",
  description: "Mini Semantic Search + Knowledge Graph Retrieval-Augmented Generation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `console.log("%c Powered by Mini-GraphRAG by HuyNNK [CC BY-NC 4.0]", "color: #6366f1; font-weight: bold; font-size: 12px;");`,
          }}
        />
      </head>
      <body className="font-sans bg-[#0a0b0f] text-zinc-100 antialiased">
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-auto">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
