"""
core/state.py
=============
Holds the application-level singletons (VectorStore, RAGPipeline) that
are shared across all route handlers.

Using a dedicated state module breaks the circular import that occurs
when routes try to import directly from ``main.py``.

Usage
-----
    # In main.py — set on startup:
    from app.core import state
    state.vector_store = VectorStore(...)
    state.rag_pipeline = RAGPipeline(...)

    # In route dependencies — read:
    from app.core import state
    store = state.vector_store
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.vector_store import VectorStore
    from app.services.rag_pipeline import RAGPipeline

# These are set once during application startup in main.py lifespan.
vector_store: "VectorStore | None" = None
rag_pipeline: "RAGPipeline | None" = None
