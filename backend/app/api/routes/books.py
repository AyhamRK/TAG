"""
routes/books.py
===============
FastAPI router for book management endpoints.

Endpoints
---------
POST   /books/upload          Upload a new PDF and index it in ChromaDB.
GET    /books                 List all indexed books with metadata.
DELETE /books/{book_id}       Remove a book and all its chunks from the index.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger

from app.core.config import settings
from app.db.vector_store import VectorStore
from app.models.schemas import (
    BookDeleteResponse,
    BookInfo,
    BookUploadResponse,
    BooksListResponse,
    ErrorResponse,
)
from app.utils.pdf_processor import load_and_chunk_all_books

router = APIRouter(prefix="/books", tags=["Books"])


# ─────────────────────────────────────────────────────────────────────────────
# Dependency — shared VectorStore instance
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store() -> VectorStore:
    """FastAPI dependency that returns the application-level VectorStore.

    The store is initialised once at startup in ``main.py`` and stored
    in ``app.core.state`` to avoid circular imports.

    Returns
    -------
    VectorStore
        The shared vector store instance.
    """
    from app.core import state
    if state.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not initialised yet. Please try again shortly.",
        )
    return state.vector_store


# ─────────────────────────────────────────────────────────────────────────────
# POST /books/upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=BookUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "File is not a PDF."},
        409: {"model": ErrorResponse, "description": "Book already indexed."},
        500: {"model": ErrorResponse, "description": "Processing failed."},
    },
    summary="Upload and index a new PDF book",
)
async def upload_book(
    file: UploadFile = File(..., description="PDF file to upload and index."),
    store: VectorStore = Depends(get_vector_store),
) -> BookUploadResponse:
    """Accept a PDF upload, save it to ``data/books/``, and index it in ChromaDB.

    The file is validated to be a ``.pdf`` before saving.  If a book with
    the same name is already indexed, a 409 Conflict is returned — delete
    it first if you want to re-index.

    Parameters
    ----------
    file : UploadFile
        The uploaded PDF file.
    store : VectorStore
        Injected vector store dependency.

    Returns
    -------
    BookUploadResponse
        Confirmation with the number of chunks created.

    Raises
    ------
    HTTPException 400
        If the uploaded file is not a PDF.
    HTTPException 409
        If a book with the same name is already indexed.
    HTTPException 500
        If PDF processing or embedding fails.
    """
    # ── Validate file type ────────────────────────────────────────────
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are accepted. Received: '{filename}'",
        )

    book_id = Path(filename).stem

    # ── Check for duplicates ──────────────────────────────────────────
    if store.is_book_indexed(book_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Book '{book_id}' is already indexed. "
                "Delete it first (DELETE /books/{book_id}) before re-uploading."
            ),
        )

    # ── Save file to disk ─────────────────────────────────────────────
    save_path = settings.books_dir / filename
    settings.books_dir.mkdir(parents=True, exist_ok=True)

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved uploaded PDF: {}", save_path)
    except Exception as exc:
        logger.error("Failed to save uploaded file '{}': {}", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {exc}",
        )
    finally:
        await file.close()

    # ── Process and index ─────────────────────────────────────────────
    try:
        chunks = load_and_chunk_all_books(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        # Filter to only the newly uploaded book's chunks.
        new_chunks = [c for c in chunks if c.book_name == book_id]

        if not new_chunks:
            raise ValueError(f"No extractable text found in '{filename}'.")

        stored = store.ingest_chunks(new_chunks)
        logger.info("Indexed '{}' — {} chunks stored.", book_id, stored)

    except Exception as exc:
        # Clean up the saved file so the state stays consistent.
        save_path.unlink(missing_ok=True)
        logger.error("Indexing failed for '{}': {}", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index book: {exc}",
        )

    return BookUploadResponse(
        book_id=book_id,
        filename=filename,
        chunks_created=stored,
        message=f"'{filename}' uploaded and indexed successfully.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /books
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=BooksListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all indexed books",
)
def list_books(
    store: VectorStore = Depends(get_vector_store),
) -> BooksListResponse:
    """Return metadata for every book currently indexed in ChromaDB.

    For each book the response includes the book ID, filename, and the
    number of chunks stored.

    Parameters
    ----------
    store : VectorStore
        Injected vector store dependency.

    Returns
    -------
    BooksListResponse
        List of :class:`~app.models.schemas.BookInfo` objects plus a
        total count.
    """
    indexed_books = store.get_indexed_books()

    books: list[BookInfo] = []
    for book_id in indexed_books:
        # Count chunks belonging to this book.
        result = store.collection.get(
            where={"book_name": book_id},
            include=["metadatas"],
        )
        chunk_count = len(result["ids"])
        books.append(
            BookInfo(
                book_id=book_id,
                filename=f"{book_id}.pdf",
                chunk_count=chunk_count,
            )
        )

    logger.debug("Listed {} book(s).", len(books))
    return BooksListResponse(books=books, total=len(books))


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /books/{book_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/{book_id}",
    response_model=BookDeleteResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Book not found."},
        500: {"model": ErrorResponse, "description": "Deletion failed."},
    },
    summary="Remove a book from the index",
)
def delete_book(
    book_id: str,
    store: VectorStore = Depends(get_vector_store),
) -> BookDeleteResponse:
    """Delete all chunks for a given book from ChromaDB.

    The PDF file on disk is also removed if it exists.  This operation
    is irreversible — the book must be re-uploaded and re-indexed to
    restore it.

    Parameters
    ----------
    book_id : str
        The book identifier (PDF filename stem, e.g. ``"Book_AI"``).
    store : VectorStore
        Injected vector store dependency.

    Returns
    -------
    BookDeleteResponse
        Confirmation message with the deleted book ID.

    Raises
    ------
    HTTPException 404
        If no chunks for ``book_id`` exist in the collection.
    HTTPException 500
        If the deletion operation fails.
    """
    # ── Verify the book exists ────────────────────────────────────────
    if not store.is_book_indexed(book_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book '{book_id}' not found in the index.",
        )

    # ── Fetch all chunk IDs for this book ─────────────────────────────
    try:
        result = store.collection.get(
            where={"book_name": book_id},
            include=["metadatas"],
        )
        chunk_ids = result["ids"]

        store.collection.delete(ids=chunk_ids)
        logger.info(
            "Deleted book '{}' — {} chunk(s) removed from ChromaDB.",
            book_id,
            len(chunk_ids),
        )
    except Exception as exc:
        logger.error("Failed to delete book '{}': {}", book_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete book from index: {exc}",
        )

    # ── Remove PDF file from disk (best-effort) ───────────────────────
    pdf_path = settings.books_dir / f"{book_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
        logger.info("Removed PDF file: {}", pdf_path)

    return BookDeleteResponse(
        book_id=book_id,
        message=f"Book '{book_id}' and all its chunks have been removed.",
    )
