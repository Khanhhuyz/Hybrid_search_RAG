"""
Embedding Service
Generates vector embeddings via Ollama (nomic-embed-text).
Supports batch processing and simple in-memory caching.
"""
import hashlib
import logging
from typing import List, Optional, Dict
from functools import lru_cache

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class EmbedderService:
    """Generate embeddings using Ollama's embedding endpoint."""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model    = settings.EMBEDDING_MODEL
        self._cache: Dict[str, List[float]] = {}

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string. Cached by content hash."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        vector = await self._call_ollama(text)
        self._cache[cache_key] = vector
        return vector

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        Ollama processes embeddings one-at-a-time, so we batch sequentially.
        """
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = await self.embed_text(text)
                embeddings.append(emb)
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(texts)} chunks")
            except Exception as e:
                logger.error(f"Embedding failed for chunk {i}: {e}")
                # Use zero vector as fallback to not break the batch
                embeddings.append([0.0] * settings.EMBEDDING_DIMENSION)
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a user query (same as text embedding with nomic-embed-text)."""
        return await self.embed_text(query)

    # ─── Ollama API ───────────────────────────────────────────────────────────

    async def _call_ollama(self, text: str) -> List[float]:
        """Call Ollama /api/embeddings endpoint."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if "embedding" not in data:
            raise ValueError(f"Ollama response missing 'embedding' field: {data}")

        return data["embedding"]

    async def health_check(self) -> dict:
        """Verify Ollama is running and model is available."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                available = any(self.model in m for m in models)
                return {
                    "status": "ok" if available else "model_not_found",
                    "model": self.model,
                    "available_models": models,
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
