"""
RAG Pipeline Service
Combines semantic search + graph retrieval for GraphRAG.
Generates answers via Ollama LLM with citations, streaming (SSE), RRF re-ranking, and caching.
"""
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator

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
    """Hybrid GraphRAG pipeline combining semantic, graph retrieval, RRF re-ranking, and streaming."""

    def __init__(
        self,
        embedder: EmbedderService,
        vector_store: VectorStoreService,
        graph_builder: GraphBuilderService,
    ):
        self.embedder      = embedder
        self.vector_store  = vector_store
        self.graph_builder = graph_builder
        # In-memory query cache: {cache_key: response_dict}
        self._response_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 100

    def _get_cache_key(self, question: str, top_k: int, use_graph: bool, doc_ids: Optional[List[str]]) -> str:
        doc_str = ",".join(sorted(doc_ids)) if doc_ids else "all"
        return f"{question.strip().lower()}:{top_k}:{use_graph}:{doc_str}"

    async def _retrieve_contexts(
        self,
        question: str,
        top_k: int = 5,
        use_graph: bool = True,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Core retrieval stage using Query Expansion, Semantic Search, and Graph Traversal."""
        # 1. Embed query (auto-translate non-English to English first)
        english_query = await self._translate_to_english(question)
        query_vector = await self.embedder.embed_query(english_query)

        # 2. Semantic Search
        semantic_results = await self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Fetch extra candidate pool for RRF
            document_ids=document_ids,
        )

        # 3. Multi-hop Graph Retrieval & Query Expansion
        graph_context_items: List[Dict] = []
        graph_entity_ids: List[str] = []

        if use_graph:
            entity_ids = self.graph_builder.find_entities_in_text(question)
            graph_entity_ids = entity_ids
            if entity_ids:
                # Retrieve multi-hop depth=2 graph neighborhood
                graph_context_items = self.graph_builder.get_related_context(
                    entity_ids, depth=2
                )

        # 4. Apply Reciprocal Rank Fusion (RRF) to merge and re-rank semantic chunks
        fused_semantic_results = self._apply_rrf(semantic_results, top_k=top_k)

        # 5. Format Prompt Contexts
        semantic_ctx = self._format_semantic_context(fused_semantic_results)
        graph_ctx    = self._format_graph_context(graph_context_items)

        prompt = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            semantic_context=semantic_ctx or "No relevant document chunks found.",
            graph_context=graph_ctx or "No graph relationships found.",
            question=question,
        )

        citations = self._build_citations(fused_semantic_results)

        if fused_semantic_results and graph_context_items:
            mode = "hybrid"
        elif graph_context_items:
            mode = "graph"
        else:
            mode = "semantic"

        graph_context_meta = {
            "entities": [
                self.graph_builder.graph.nodes[eid].get("label", eid)
                for eid in graph_entity_ids
                if self.graph_builder.graph.has_node(eid)
            ],
            "relations": [{"text": item["text"]} for item in graph_context_items[:5]],
        }

        return {
            "prompt": prompt,
            "citations": citations,
            "graph_context": graph_context_meta,
            "semantic_chunks_used": len(fused_semantic_results),
            "graph_nodes_used": len(graph_entity_ids),
            "retrieval_mode": mode,
        }

    def _apply_rrf(self, semantic_results: List[Dict], top_k: int = 5, k_constant: int = 60) -> List[Dict]:
        """Apply Reciprocal Rank Fusion (RRF) algorithm to re-rank search results."""
        if not semantic_results:
            return []

        # Calculate RRF score for each result
        scored_results = []
        for rank, r in enumerate(semantic_results):
            # Reciprocal rank score
            rrf_score = 1.0 / (k_constant + (rank + 1))
            # Blend with original cosine similarity score if available
            orig_score = r.get("score", 0.0)
            combined_score = rrf_score + (0.5 * orig_score)
            
            r_copy = dict(r)
            r_copy["rrf_score"] = combined_score
            scored_results.append(r_copy)

        # Sort by combined score descending
        scored_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored_results[:top_k]

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        use_graph: bool = True,
        document_ids: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Full GraphRAG pipeline with caching."""
        cache_key = self._get_cache_key(question, top_k, use_graph, document_ids)
        if cache_key in self._response_cache:
            logger.info(f"Returning cached answer for query: '{question}'")
            return self._response_cache[cache_key]

        retrieval = await self._retrieve_contexts(question, top_k, use_graph, document_ids)
        answer_text = await self._call_llm(retrieval["prompt"])

        result = {
            "question":             question,
            "answer":               answer_text.strip(),
            "citations":            retrieval["citations"],
            "graph_context":        retrieval["graph_context"],
            "semantic_chunks_used": retrieval["semantic_chunks_used"],
            "graph_nodes_used":     retrieval["graph_nodes_used"],
            "model_used":           settings.LLM_MODEL,
            "retrieval_mode":       retrieval["retrieval_mode"],
        }

        # Cache result
        if len(self._response_cache) >= self._cache_max_size:
            # Drop oldest key
            oldest = next(iter(self._response_cache))
            del self._response_cache[oldest]
        self._response_cache[cache_key] = result

        return result

    async def answer_stream(
        self,
        question: str,
        top_k: int = 5,
        use_graph: bool = True,
        document_ids: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming GraphRAG response via SSE (Server-Sent Events).
        First yields metadata event, followed by streaming token events.
        """
        retrieval = await self._retrieve_contexts(question, top_k, use_graph, document_ids)

        # 1. Send metadata payload first
        meta_payload = {
            "type": "metadata",
            "data": {
                "question": question,
                "citations": retrieval["citations"],
                "graph_context": retrieval["graph_context"],
                "semantic_chunks_used": retrieval["semantic_chunks_used"],
                "graph_nodes_used": retrieval["graph_nodes_used"],
                "model_used": settings.LLM_MODEL,
                "retrieval_mode": retrieval["retrieval_mode"],
                "engine_origin": "quinc-fptu/mini-graphrag:cc-by-nc-4.0",
            }
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        # 2. Stream tokens from Ollama
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": retrieval["prompt"],
            "stream": True,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }

        full_answer = []
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                full_answer.append(chunk)
                                token_event = {
                                    "type": "token",
                                    "data": {"text": chunk}
                                }
                                yield f"data: {json.dumps(token_event)}\n\n"
                        except json.JSONDecodeError:
                            continue

            # 3. Send done event
            done_event = {"type": "done", "data": {}}
            yield f"data: {json.dumps(done_event)}\n\n"

            # Save in cache
            cache_key = self._get_cache_key(question, top_k, use_graph, document_ids)
            cached_result = {
                "question": question,
                "answer": "".join(full_answer).strip(),
                "citations": retrieval["citations"],
                "graph_context": retrieval["graph_context"],
                "semantic_chunks_used": retrieval["semantic_chunks_used"],
                "graph_nodes_used": retrieval["graph_nodes_used"],
                "model_used": settings.LLM_MODEL,
                "retrieval_mode": retrieval["retrieval_mode"],
            }
            self._response_cache[cache_key] = cached_result

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            err_event = {"type": "error", "data": {"error": str(e)}}
            yield f"data: {json.dumps(err_event)}\n\n"

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

