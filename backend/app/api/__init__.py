"""
API package init — export all routers.
"""
from app.api import documents, search, graph

__all__ = ["documents", "search", "graph"]
