"""Sparse retrieval, true reciprocal-rank fusion, and optional reranking."""
from __future__ import annotations

import asyncio
import logging
import math
import re
import unicodedata
import json
from collections import Counter, defaultdict
from typing import Dict, List, Sequence
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Unicode-aware word and identifier tokenizer suitable for Vietnamese."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return re.findall(r"[\w.-]+", normalized, flags=re.UNICODE)


class BM25Retriever:
    """Small in-process BM25 index built from the canonical chunk database."""

    def __init__(self, k1: float | None = None, b: float | None = None):
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B

    def search(self, query: str, chunks: Sequence[Dict], top_k: int) -> List[Dict]:
        if not chunks:
            return []
        documents = [tokenize(item.get("content", "")) for item in chunks]
        avg_len = sum(map(len, documents)) / max(len(documents), 1)
        df: Counter = Counter()
        for terms in documents:
            df.update(set(terms))
        query_terms = tokenize(query)
        results = []
        for item, terms in zip(chunks, documents):
            tf = Counter(terms)
            score = 0.0
            for term in query_terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(documents) - df[term] + 0.5) / (df[term] + 0.5))
                norm = frequency + self.k1 * (1 - self.b + self.b * len(terms) / max(avg_len, 1))
                score += idf * frequency * (self.k1 + 1) / norm
            if score > 0:
                result = dict(item)
                result["sparse_score"] = score
                result["score"] = score
                result["retriever"] = "sparse"
                results.append(result)
        return sorted(results, key=lambda row: row["sparse_score"], reverse=True)[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, Sequence[Dict]],
    weights: Dict[str, float] | None = None,
    k: int | None = None,
) -> List[Dict]:
    """Fuse independent ranked lists by chunk id without mixing score scales."""
    weights = weights or {}
    k = k or settings.RRF_K
    scores = defaultdict(float)
    records: Dict[str, Dict] = {}
    sources = defaultdict(list)
    for name, items in ranked_lists.items():
        weight = weights.get(name, 1.0)
        for rank, item in enumerate(items, 1):
            item_id = str(item.get("id") or item.get("chunk_id") or "")
            if not item_id:
                continue
            scores[item_id] += weight / (k + rank)
            records.setdefault(item_id, dict(item))
            sources[item_id].append(name)
    output = []
    for item_id, score in scores.items():
        row = records[item_id]
        row["rrf_score"] = score
        row["retrieval_sources"] = sources[item_id]
        output.append(row)
    return sorted(output, key=lambda row: row["rrf_score"], reverse=True)


class MultilingualReranker:
    """Cross-encoder reranker when configured, deterministic lexical fallback otherwise."""

    def __init__(self):
        self._model = None
        self._load_attempted = False

    def _load(self):
        if self._load_attempted or not settings.RERANKER_MODEL:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(settings.RERANKER_MODEL)
        except Exception as exc:  # optional dependency/model
            logger.warning("Cross-encoder unavailable; using multilingual lexical reranker: %s", exc)

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        q, d = Counter(tokenize(query)), Counter(tokenize(text))
        if not q or not d:
            return 0.0
        overlap = sum(min(count, d.get(term, 0)) for term, count in q.items())
        coverage = overlap / sum(q.values())
        phrase = 1.0 if unicodedata.normalize("NFKC", query).casefold() in unicodedata.normalize("NFKC", text).casefold() else 0.0
        return min(1.0, 0.85 * coverage + 0.15 * phrase)

    async def _ollama_scores(self, query: str, candidates: Sequence[Dict]) -> List[float] | None:
        """Use the existing multilingual LLM as a zero-shot relevance reranker."""
        scores: List[float] = []
        size = max(1, settings.RERANKER_OLLAMA_BATCH_SIZE)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for offset in range(0, len(candidates), size):
                    batch = candidates[offset:offset + size]
                    passages = "\n".join(
                        f"[{index}] {row.get('content', '')[:1200]}"
                        for index, row in enumerate(batch)
                    )
                    prompt = (
                        "Score how well each passage answers the query, regardless of language. "
                        "Use a number from 0 (irrelevant) to 1 (direct evidence). Return only JSON "
                        "in the form {\"scores\":[0.0]}; preserve passage order.\n\n"
                        f"Query: {query}\n\nPassages:\n{passages}"
                    )
                    response = await client.post(
                        f"{settings.OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": settings.LLM_MODEL,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": 0.0, "num_predict": 128},
                        },
                    )
                    response.raise_for_status()
                    payload = json.loads(response.json().get("response", "{}"))
                    batch_scores = payload.get("scores", [])
                    if len(batch_scores) != len(batch):
                        return None
                    scores.extend(max(0.0, min(1.0, float(value))) for value in batch_scores)
            return scores
        except Exception as exc:
            logger.warning("Ollama reranker unavailable; using lexical fallback: %s", exc)
            return None

    async def rerank(self, query: str, candidates: Sequence[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []
        if settings.RERANKER_ENABLED:
            await asyncio.to_thread(self._load)
        if self._model:
            pairs = [(query, row.get("content", "")) for row in candidates]
            raw_scores = await asyncio.to_thread(self._model.predict, pairs)
            scores = [1 / (1 + math.exp(-float(value))) for value in raw_scores]
        else:
            scores = await self._ollama_scores(query, candidates) if settings.RERANKER_OLLAMA_FALLBACK else None
            if scores is None:
                scores = [self._lexical_score(query, row.get("content", "")) for row in candidates]
        ranked = []
        for row, score in zip(candidates, scores):
            item = dict(row)
            item["rerank_score"] = score
            # Keep a bounded evidence score for no-answer decisions/calibration.
            item["evidence_score"] = 0.8 * score + 0.2 * min(1.0, item.get("rrf_score", 0.0) * settings.RRF_K)
            ranked.append(item)
        ranked.sort(key=lambda item: item["evidence_score"], reverse=True)
        filtered = [item for item in ranked if item["evidence_score"] >= settings.RERANK_MIN_SCORE]
        return (filtered or ranked[:1])[:top_k]
