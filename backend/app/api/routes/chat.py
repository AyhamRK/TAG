"""
routes/chat.py
==============
FastAPI router for the question-answering chat endpoint.

Endpoints
---------
POST /chat/ask    Run a question through the guardrail then the RAG pipeline.

Flow
----
1. Validate the request body (Pydantic).
2. Pass the question to the guardrail.
   a. If rejected → return 200 with ``allowed=False`` and the refusal message.
   b. If allowed  → proceed.
3. Optionally filter retrieved chunks to a specific book (``book_id``).
4. Run the RAG pipeline and return the answer with sources.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    GuardrailRejectionResponse,
    SourceChunk,
)
from app.services.guardrail import Guardrail
from app.services.rag_pipeline import RAGPipeline
from app.db.vector_store import VectorStore

router = APIRouter(prefix="/chat", tags=["Chat"])


# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store() -> VectorStore:
    """Return the application-level VectorStore instance.

    Returns
    -------
    VectorStore
        Shared vector store initialised at startup.

    Raises
    ------
    HTTPException 503
        If the store has not been initialised yet.
    """
    from app.core import state
    if state.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not ready. Please try again shortly.",
        )
    return state.vector_store


def get_rag_pipeline() -> RAGPipeline:
    """Return the application-level RAGPipeline instance.

    Returns
    -------
    RAGPipeline
        Shared pipeline initialised at startup.

    Raises
    ------
    HTTPException 503
        If the pipeline has not been initialised yet.
    """
    from app.core import state
    if state.rag_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not ready. Please try again shortly.",
        )
    return state.rag_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat/ask
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": (
                "Either a successful answer (ChatResponse) "
                "or a guardrail rejection (GuardrailRejectionResponse)."
            ),
        },
        422: {"description": "Validation error — invalid request body."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
        503: {"model": ErrorResponse, "description": "Pipeline not ready."},
    },
    summary="Ask a question about the books",
)
def ask_question(
    request: ChatRequest,
    store: VectorStore = Depends(get_vector_store),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse | GuardrailRejectionResponse:
    """Answer a question using the RAG pipeline, gated by the topic guardrail.

    The endpoint runs in two stages:

    **Stage 1 — Guardrail check**
    The question is embedded and compared against all stored book chunks.
    If the best cosine similarity score is below the configured threshold
    (default 0.30), or if the question is a greeting/small talk, it is
    rejected immediately and a :class:`GuardrailRejectionResponse` is
    returned with HTTP 200.

    **Stage 2 — RAG answer generation**
    If the question passes the guardrail, the top-K most relevant chunks
    are passed to Flan-T5-Large with a strict *answer-only-from-context*
    prompt.  The answer and source citations are returned as a
    :class:`ChatResponse`.

    If ``book_id`` is provided in the request, only chunks from that
    specific book are eligible for retrieval.

    Parameters
    ----------
    request : ChatRequest
        Contains the ``question`` (required) and optional ``book_id``.
    store : VectorStore
        Injected vector store dependency.
    pipeline : RAGPipeline
        Injected RAG pipeline dependency.

    Returns
    -------
    ChatResponse
        When the question is allowed and an answer is generated.
    GuardrailRejectionResponse
        When the question is off-topic or is small talk.

    Raises
    ------
    HTTPException 500
        If an unexpected error occurs during generation.
    HTTPException 503
        If the pipeline or vector store is not initialised.
    """
    question = request.question.strip()
    logger.info("Chat request — question: '{}', book_id: {}", question[:80], request.book_id)

    # ── Stage 1: Guardrail ────────────────────────────────────────────
    from app.core.config import settings
    guardrail = Guardrail(
        store,
        threshold=settings.topic_similarity_threshold,
        top_k=settings.retriever_top_k,
    )

    guard_result = guardrail.check(question)

    if not guard_result.allowed:
        logger.info(
            "Guardrail rejected question (score={:.3f}): '{}'",
            guard_result.best_score,
            question[:60],
        )
        return GuardrailRejectionResponse(
            allowed=False,
            message=guard_result.message,
        )

    # ── Stage 2: Optional book filter ────────────────────────────────
    chunks = guard_result.chunks

    if request.book_id:
        filtered = [c for c in chunks if c.get("book_name") == request.book_id]
        if not filtered:
            # The book exists but has no relevant chunks for this question.
            # Fall back to unfiltered results rather than returning empty.
            logger.warning(
                "book_id filter '{}' yielded no chunks — using unfiltered results.",
                request.book_id,
            )
        else:
            chunks = filtered
            logger.debug(
                "Filtered to {} chunk(s) from book '{}'.",
                len(chunks),
                request.book_id,
            )

    # ── Stage 3: RAG generation ───────────────────────────────────────
    try:
        rag_response = pipeline.answer(question)
    except RuntimeError as exc:
        logger.error("RAG pipeline error for question '{}': {}", question[:60], exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Unexpected error in RAG pipeline: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the answer.",
        )

    # ── Build source citations ────────────────────────────────────────
    sources = [
        SourceChunk(
            page=src.page_number,
            book=f"{src.book_name}.pdf",
            chunk=src.text_snippet[:300],
            score=src.score,
        )
        for src in rag_response.sources
    ]

    logger.info(
        "Answer generated for '{}…' — {} source(s) cited.",
        question[:50],
        len(sources),
    )

    return ChatResponse(
        answer=rag_response.answer,
        sources=sources,
        question=question,
        model_used=rag_response.model_used,
    )
