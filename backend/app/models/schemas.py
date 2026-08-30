"""
models/schemas.py
=================
Pydantic v2 request and response models for all API endpoints.

These models are used by FastAPI for:
* Automatic request body validation and parsing
* Response serialisation and OpenAPI schema generation
* Swagger UI documentation

Import individual models as needed:

    from app.models.schemas import ChatRequest, ChatResponse, BookInfo
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Shared / reusable models
# ─────────────────────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """A single source chunk cited in an answer.

    Attributes
    ----------
    page : int
        1-based page number within the source PDF.
    book : str
        Human-readable book name (PDF filename stem).
    chunk : str
        The raw text of the retrieved chunk (truncated to 300 chars for
        readability in responses).
    score : float
        Cosine similarity score between the query and this chunk
        (0.0 – 1.0, higher is more relevant).
    """

    page: int = Field(..., description="Page number in the source PDF (1-based).")
    book: str = Field(..., description="Book name (PDF filename without extension).")
    chunk: str = Field(..., description="Excerpt from the retrieved chunk.")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0–1).")


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for POST /chat/ask.

    Attributes
    ----------
    question : str
        The user's natural-language question (1–2000 characters).
    book_id : Optional[str]
        If provided, restrict retrieval to this specific book (filename
        stem, e.g. ``"Book_AI"``).  If ``None``, all books are searched.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask the assistant.",
        examples=["What is the knowledge economy?"],
    )
    book_id: Optional[str] = Field(
        default=None,
        description="Optional book name to restrict the search to.",
        examples=["The_Brave_Knowledge_World"],
    )


class ChatResponse(BaseModel):
    """Successful response body for POST /chat/ask.

    Attributes
    ----------
    answer : str
        The model-generated answer grounded in the book content.
    sources : List[SourceChunk]
        The chunks used to generate the answer, ordered by relevance.
    question : str
        Echo of the original question (useful for UI display).
    model_used : str
        Which model produced the answer (``"flan-t5"`` or ``"bart"``).
    """

    answer: str = Field(..., description="Generated answer from the books.")
    sources: List[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer.",
    )
    question: str = Field(..., description="The original question.")
    model_used: str = Field(
        default="flan-t5",
        description="Model that generated the answer: 'flan-t5' or 'bart'.",
    )


class GuardrailRejectionResponse(BaseModel):
    """Response body when the guardrail rejects a question.

    Attributes
    ----------
    allowed : bool
        Always ``False`` for this response type.
    message : str
        Human-readable refusal message shown to the user.
    """

    allowed: bool = Field(default=False)
    message: str = Field(
        ...,
        description="Reason the question was rejected.",
        examples=["I can only answer questions related to Talal Abu Ghazaleh's books."],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Books
# ─────────────────────────────────────────────────────────────────────────────

class BookInfo(BaseModel):
    """Metadata about a single indexed book.

    Attributes
    ----------
    book_id : str
        Unique identifier (PDF filename stem, e.g. ``"Book_AI"``).
    filename : str
        Full filename including extension (e.g. ``"Book_AI.pdf"``).
    chunk_count : int
        Number of chunks currently stored in ChromaDB for this book.
    """

    book_id: str = Field(..., description="Unique book identifier (filename stem).")
    filename: str = Field(..., description="PDF filename with extension.")
    chunk_count: int = Field(..., ge=0, description="Number of indexed chunks.")


class BooksListResponse(BaseModel):
    """Response body for GET /books.

    Attributes
    ----------
    books : List[BookInfo]
        Metadata for every indexed book.
    total : int
        Total number of indexed books.
    """

    books: List[BookInfo] = Field(default_factory=list)
    total: int = Field(..., ge=0, description="Total number of books.")


class BookUploadResponse(BaseModel):
    """Response body for POST /books/upload.

    Attributes
    ----------
    book_id : str
        Identifier of the newly indexed book.
    filename : str
        Saved filename.
    chunks_created : int
        Number of chunks stored in ChromaDB.
    message : str
        Human-readable status message.
    """

    book_id: str = Field(..., description="Identifier of the uploaded book.")
    filename: str = Field(..., description="Saved PDF filename.")
    chunks_created: int = Field(..., ge=0, description="Chunks added to the index.")
    message: str = Field(..., description="Status message.")


class BookDeleteResponse(BaseModel):
    """Response body for DELETE /books/{book_id}.

    Attributes
    ----------
    book_id : str
        The identifier of the deleted book.
    message : str
        Confirmation message.
    """

    book_id: str = Field(..., description="Identifier of the deleted book.")
    message: str = Field(..., description="Confirmation message.")


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health.

    Attributes
    ----------
    status : str
        ``"ok"`` when the service is healthy.
    indexed_books : List[str]
        Names of all currently indexed books.
    total_chunks : int
        Total number of chunks in the vector store.
    version : str
        Application version string.
    """

    status: str = Field(default="ok")
    indexed_books: List[str] = Field(default_factory=list)
    total_chunks: int = Field(default=0)
    version: str = Field(default="1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# Generic error
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Generic error response body.

    Attributes
    ----------
    detail : str
        Human-readable description of what went wrong.
    """

    detail: str = Field(..., description="Error description.")
