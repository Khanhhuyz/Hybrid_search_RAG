"""
RAG Pipeline Service
Combines semantic search + graph retrieval for GraphRAG.
Generates answers via Ollama LLM with citations.
"""
import logging
from typing import List, Dict, Any, Optional

import httpx

from app.config import settings
from app.services.embedder import EmbedderService
from app.services.vector_store import VectorStoreService
from app.services.graph_builder import GraphBuilderService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable AI assistant that answers questions based ONLY on the provided context.
Your answers should be:
- Accurate and grounded in the context
- Clear and concise
- Reference the source documents when relevant
- Honest about uncertainty if the context does not contain the answer
- Answer in the SAME language as the user's question (e.g. if the question is in Vietnamese, you MUST reply in Vietnamese, even if the context documents are in English).

If the context does not contain enough information to answer the question, say so clearly."""

RAG_PROMPT_TEMPLATE = """{system}

=== DOCUMENT CONTEXT ===
{semantic_context}

=== KNOWLEDGE GRAPH CONTEXT ===
{graph_context}

=== QUESTION ===
{question}

=== ANSWER ==="""


class RAGPipeline:
    """Hybrid GraphRAG pipeline combining semantic and graph retrieval."""

    def __init__(
        self,
        embedder: EmbedderService,
        vector_store: VectorStoreService,
        graph_builder: GraphBuilderService,
    ):
        self.embedder      = embedder
        self.vector_store  = vector_store
        self.graph_builder = graph_builder

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        use_graph: bool = True,
        document_ids: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Full GraphRAG pipeline:
        1. Embed the question
        2. Semantic search → top-K chunks
        3. Graph entity detection + neighborhood traversal
        4. Merge context → build prompt
        5. LLM generation
        6. Return answer with citations
        """
        # ── Step 1: Embed query (auto-translate non-English to English first) ──
        english_query = await self._translate_to_english(question)
        query_vector = await self.embedder.embed_query(english_query)

        # ── Step 2: Semantic Search ──────────────────────────────────────────
        semantic_results = await self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            document_ids=document_ids,
        )

        # ── Step 3: Graph Retrieval ──────────────────────────────────────────
        graph_context_items: List[Dict] = []
        graph_entity_ids: List[str] = []

        if use_graph:
            entity_ids = self.graph_builder.find_entities_in_text(question)
            graph_entity_ids = entity_ids
            if entity_ids:
                graph_context_items = self.graph_builder.get_related_context(
                    entity_ids, depth=1
                )

        # ── Step 4: Build Prompt ─────────────────────────────────────────────
        semantic_ctx = self._format_semantic_context(semantic_results)
        graph_ctx    = self._format_graph_context(graph_context_items)
        prompt       = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            semantic_context=semantic_ctx or "No relevant document chunks found.",
            graph_context=graph_ctx or "No graph relationships found.",
            question=question,
        )

        # ── Step 5: LLM Generation ───────────────────────────────────────────
        answer_text = await self._call_llm(prompt)

        # ── Step 6: Build Citations ───────────────────────────────────────────
        citations = self._build_citations(semantic_results)

        # ── Determine retrieval mode ─────────────────────────────────────────
        if semantic_results and graph_context_items:
            mode = "hybrid"
        elif graph_context_items:
            mode = "graph"
        else:
            mode = "semantic"

        return {
            "question":             question,
            "answer":               answer_text.strip(),
            "citations":            citations,
            "graph_context":        {
                "entities": [self.graph_builder.graph.nodes[eid].get("label", eid) for eid in graph_entity_ids if self.graph_builder.graph.has_node(eid)],
                "relations": [{"text": item["text"]} for item in graph_context_items[:5]],
            },
            "semantic_chunks_used": len(semantic_results),
            "graph_nodes_used":     len(graph_entity_ids),
            "model_used":           settings.LLM_MODEL,
            "retrieval_mode":       mode,
        }

    # ─── Semantic Search (standalone) ────────────────────────────────────────

    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Pure semantic similarity search (with auto query translation)."""
        english_query = await self._translate_to_english(query)
        query_vector = await self.embedder.embed_query(english_query)
        return await self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            document_ids=document_ids,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _format_semantic_context(self, results: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(results[:settings.MAX_CONTEXT_CHUNKS]):
            src  = r.get("document_filename", "Unknown")
            page = r.get("page_number", "?")
            text = r.get("content", "")[:800]
            parts.append(f"[{i+1}] Source: {src} (Page {page})\n{text}")
        return "\n\n".join(parts)

    def _format_graph_context(self, items: List[Dict]) -> str:
        if not items:
            return ""
        return "\n".join(f"• {item['text']}" for item in items)

    def _build_citations(self, results: List[Dict]) -> List[Dict]:
        citations = []
        seen = set()
        for r in results:
            chunk_id = r.get("id")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            citations.append({
                "document_id":       r.get("document_id"),
                "document_filename": r.get("document_filename", "Unknown"),
                "chunk_id":          chunk_id,
                "chunk_index":       r.get("chunk_index", 0),
                "page_number":       r.get("page_number"),
                "relevance_score":   round(r.get("score", 0), 4),
                "excerpt":           r.get("content", "")[:200] + "...",
            })
        return citations

    async def _call_llm(self, prompt: str) -> str:
        """Generate answer via Ollama /api/generate."""
        payload = {
            "model":  settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def _translate_to_english(self, text: str) -> str:
        """
        Detect if text is non-English and translate to English for better embedding.
        Uses a fast single-pass LLM call. Falls back to original text on any error.
        """
        # Quick heuristic: if all chars are ASCII printable, assume already English
        try:
            text.encode("ascii")
            return text  # Pure ASCII → probably English, skip translation
        except UnicodeEncodeError:
            pass  # Contains non-ASCII → proceed to translate

        try:
            prompt = (
                "Translate the following text to English. "
                "Output ONLY the English translation, nothing else. "
                "If the text is already in English, output it unchanged.\n\n"
                f"Text: {text}\n\nEnglish translation:"
            )
            payload = {
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                translated = resp.json().get("response", "").strip()
                if translated:
                    logger.info(f"Query translated: '{text}' → '{translated}'")
                    return translated
        except Exception as e:
            logger.warning(f"Translation failed, using original query: {e}")

        return text
