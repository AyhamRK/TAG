"""
main.py
=======
FastAPI application entry point.

Startup sequence
----------------
1. Configure loguru logging.
2. Scan ``data/books/`` and ingest any new PDFs into ChromaDB.
3. Initialise the shared VectorStore and RAGPipeline, stored in
   ``app.core.state`` for access by route dependencies.
4. Register API routers.
5. Begin serving requests.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core import state
from app.db.vector_store import VectorStore
from app.services.rag_pipeline import RAGPipeline
from app.utils.pdf_processor import load_and_chunk_all_books
from app.models.schemas import HealthResponse


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    logger.info("Starting up {} v{}", settings.app_title, settings.app_version)

    # Ingest new PDFs
    try:
        chunks = load_and_chunk_all_books(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    except FileNotFoundError as exc:
        logger.warning("Books directory not found: {}", exc)
        chunks = []

    # Initialise singletons and store in state module
    state.vector_store = VectorStore(
        chroma_dir=settings.chroma_db_dir,
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model_name,
        device=settings.embedding_device,
    )

    if chunks:
        state.vector_store.ingest_chunks(chunks)

    state.rag_pipeline = RAGPipeline(
        vector_store=state.vector_store,
        model_name=settings.llm_model_name,
        summariser_model_name=settings.summariser_model_name,
        device=settings.llm_device,
        max_new_tokens=settings.llm_max_new_tokens,
        top_k=settings.retriever_top_k,
    )

    logger.info(
        "Ready — {} book(s), {} chunk(s). Swagger: http://{}:{}/docs",
        len(state.vector_store.get_indexed_books()),
        state.vector_store.count(),
        settings.app_host,
        settings.app_port,
    )

    yield

    logger.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="Local RAG assistant — answers questions from Talal Abu Ghazaleh's books.",
    lifespan=lifespan,
    debug=settings.app_debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.routes import books, chat  # noqa: E402

app.include_router(books.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Service health check")
def health_check() -> HealthResponse:
    """Return service health, indexed books, and chunk count."""
    return HealthResponse(
        status="ok",
        indexed_books=state.vector_store.get_indexed_books() if state.vector_store else [],
        total_chunks=state.vector_store.count() if state.vector_store else 0,
        version=settings.app_version,
    )
