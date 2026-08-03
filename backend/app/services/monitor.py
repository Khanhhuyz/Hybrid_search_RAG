"""
Monitor Service
Runtime monitoring and observability:
- Per-stage latency tracking
- Query statistics
- Error rate tracking
- Token usage estimation
"""
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """A single query log entry."""
    timestamp: str
    question: str
    query_type: str  # local, global, hybrid
    retrieval_mode: str
    timings_ms: Dict[str, float]
    semantic_chunks_used: int
    graph_nodes_used: int
    confidence_score: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class Monitor:
    """Runtime monitoring and query statistics."""

    def __init__(self, max_log_size: int = None):
        self.max_log_size = max_log_size or settings.MAX_QUERY_LOG_SIZE
        self._query_logs: deque = deque(maxlen=self.max_log_size)
        self._error_count: int = 0
        self._total_queries: int = 0
        self._total_latency_ms: float = 0.0
        self._stage_latencies: Dict[str, List[float]] = {}

    # ─── Timer Context Manager ───────────────────────────────────────────────

    class Timer:
        """Context manager for timing code blocks."""
        def __init__(self, monitor: "Monitor", stage_name: str, timings: Dict[str, float]):
            self.monitor = monitor
            self.stage_name = stage_name
            self.timings = timings
            self._start = 0.0

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self.timings[self.stage_name] = round(elapsed_ms, 2)
            self.monitor._record_stage_latency(self.stage_name, elapsed_ms)

    def timer(self, stage_name: str, timings: Dict[str, float]) -> "Timer":
        """Create a timer for a specific pipeline stage."""
        return self.Timer(self, stage_name, timings)

    # ─── Query Logging ───────────────────────────────────────────────────────

    def log_query(
        self,
        question: str,
        query_type: str,
        retrieval_mode: str,
        timings_ms: Dict[str, float],
        semantic_chunks_used: int = 0,
        graph_nodes_used: int = 0,
        confidence_score: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log a completed query."""
        entry = QueryLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            question=question[:200],  # Truncate for storage
            query_type=query_type,
            retrieval_mode=retrieval_mode,
            timings_ms=timings_ms,
            semantic_chunks_used=semantic_chunks_used,
            graph_nodes_used=graph_nodes_used,
            confidence_score=confidence_score,
            success=success,
            error=error,
        )

        self._query_logs.append(entry)
        self._total_queries += 1

        total_ms = sum(timings_ms.values())
        self._total_latency_ms += total_ms

        if not success:
            self._error_count += 1

        if settings.DEBUG:
            logger.debug(f"Query logged: type={query_type}, total_ms={total_ms:.1f}")

    def _record_stage_latency(self, stage: str, latency_ms: float):
        """Record latency for a specific stage."""
        if stage not in self._stage_latencies:
            self._stage_latencies[stage] = []
        self._stage_latencies[stage].append(latency_ms)
        # Keep only last 100 entries per stage
        if len(self._stage_latencies[stage]) > 100:
            self._stage_latencies[stage] = self._stage_latencies[stage][-100:]

    # ─── Statistics ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        avg_latency = (
            self._total_latency_ms / self._total_queries
            if self._total_queries > 0
            else 0
        )

        # Query type distribution
        type_counts: Dict[str, int] = {}
        mode_counts: Dict[str, int] = {}
        for log in self._query_logs:
            type_counts[log.query_type] = type_counts.get(log.query_type, 0) + 1
            mode_counts[log.retrieval_mode] = mode_counts.get(log.retrieval_mode, 0) + 1

        # Per-stage latency stats
        stage_stats = {}
        for stage, latencies in self._stage_latencies.items():
            if latencies:
                stage_stats[stage] = {
                    "avg_ms": round(sum(latencies) / len(latencies), 2),
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "count": len(latencies),
                }

        # Average confidence
        confidences = [log.confidence_score for log in self._query_logs if log.success]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_queries": self._total_queries,
            "error_count": self._error_count,
            "error_rate": round(
                self._error_count / self._total_queries if self._total_queries > 0 else 0, 4
            ),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_confidence": round(avg_confidence, 3),
            "query_type_distribution": type_counts,
            "retrieval_mode_distribution": mode_counts,
            "stage_latencies": stage_stats,
            "recent_queries_count": len(self._query_logs),
        }

    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """Get recent query logs."""
        recent = list(self._query_logs)[-limit:]
        return [q.to_dict() for q in reversed(recent)]
