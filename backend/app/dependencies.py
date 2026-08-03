"""
Dependency injection for shared services.
Services are singletons instantiated once on startup.
"""
from app.services.embedder import EmbedderService
from app.services.vector_store import VectorStoreService
from app.services.graph_builder import GraphBuilderService
from app.services.rag_pipeline import RAGPipeline
from app.services.monitor import Monitor

# Singleton instances
_embedder: EmbedderService       = None
_vector_store: VectorStoreService = None
_graph_builder: GraphBuilderService = None
_rag_pipeline: RAGPipeline        = None
_monitor: Monitor                  = None


def init_services():
    """Initialize all services. Called on application startup."""
    global _embedder, _vector_store, _graph_builder, _rag_pipeline, _monitor
    _monitor       = Monitor()
    _embedder      = EmbedderService()
    _vector_store  = VectorStoreService()
    _graph_builder = GraphBuilderService()
    _rag_pipeline  = RAGPipeline(_embedder, _vector_store, _graph_builder, _monitor)


def get_embedder() -> EmbedderService:
    return _embedder


def get_vector_store() -> VectorStoreService:
    return _vector_store


def get_graph_builder() -> GraphBuilderService:
    return _graph_builder


def get_rag_pipeline() -> RAGPipeline:
    return _rag_pipeline


def get_monitor() -> Monitor:
    return _monitor
