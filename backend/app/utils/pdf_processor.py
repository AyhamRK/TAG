"""
pdf_processor.py
================
Responsible for discovering, parsing, and chunking all PDF books stored in
the local ``data/books/`` directory.

No file paths are accepted as arguments — the module scans the books folder
automatically and processes every ``.pdf`` file it finds.

Typical usage
-------------
    from app.utils.pdf_processor import load_and_chunk_all_books

    chunks = load_and_chunk_all_books()
    for chunk in chunks:
        print(chunk["book_name"], chunk["page_number"], chunk["text"][:80])
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from loguru import logger

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Resolved once at import time so every call uses the same path.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]          # …/backend/
BOOKS_DIR: Path = _PROJECT_ROOT / "data" / "books"

DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))


# ─────────────────────────────────────────────────────────────────────────────
# Public data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single text chunk extracted from a PDF page.

    Attributes
    ----------
    text : str
        The raw text content of the chunk (stripped of leading/trailing
        whitespace).
    page_number : int
        1-based page number within the source PDF where the chunk begins.
    source : str
        Absolute path to the originating PDF file as a string.
    book_name : str
        Human-readable book title derived from the PDF filename
        (stem only, e.g. ``"knowledge_economy"``).
    chunk_index : int
        Sequential index of this chunk within its page's text.
    """

    text: str
    page_number: int
    source: str
    book_name: str
    chunk_index: int

    def to_dict(self) -> dict:
        """Return a plain dictionary representation of this chunk.

        Returns
        -------
        dict
            Keys: ``text``, ``page_number``, ``source``, ``book_name``,
            ``chunk_index``.
        """
        return {
            "text": self.text,
            "page_number": self.page_number,
            "source": self.source,
            "book_name": self.book_name,
            "chunk_index": self.chunk_index,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _discover_pdfs(books_dir: Path) -> List[Path]:
    """Return a sorted list of all ``.pdf`` files inside *books_dir*.

    Non-PDF files and subdirectories are silently skipped.

    Parameters
    ----------
    books_dir : Path
        Directory to scan.  Must exist.

    Returns
    -------
    List[Path]
        Sorted list of ``.pdf`` ``Path`` objects found directly in
        *books_dir* (non-recursive).

    Raises
    ------
    FileNotFoundError
        If *books_dir* does not exist.
    """
    if not books_dir.exists():
        raise FileNotFoundError(
            f"Books directory not found: {books_dir}\n"
            "Create the directory and place your PDF files inside it."
        )

    pdf_files = sorted(
        p for p in books_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )

    skipped = [
        p.name for p in books_dir.iterdir()
        if p.is_file() and p.suffix.lower() != ".pdf"
    ]
    if skipped:
        logger.warning(
            "Skipped {} non-PDF file(s) in {}: {}",
            len(skipped),
            books_dir,
            ", ".join(skipped),
        )

    return pdf_files


def _extract_pages(pdf_path: Path) -> List[tuple[int, str]]:
    """Extract raw text from every page of a PDF using PyMuPDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    List[tuple[int, str]]
        List of ``(page_number, page_text)`` tuples.  *page_number* is
        1-based.  Pages that yield no extractable text are excluded.
    """
    pages: List[tuple[int, str]] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Could not open PDF {}: {}", pdf_path.name, exc)
        return pages

    logger.debug("Extracting text from '{}' ({} pages)…", pdf_path.name, len(doc))

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")          # plain UTF-8 text layer

        # Collapse runs of blank lines and strip edge whitespace.
        text = "\n".join(
            line for line in text.splitlines() if line.strip()
        ).strip()

        if text:
            pages.append((page_index + 1, text))   # convert to 1-based

    doc.close()
    logger.debug(
        "  '{}': {} page(s) contained extractable text.",
        pdf_path.name,
        len(pages),
    )
    return pages


def _split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """Split a string into overlapping fixed-size character chunks.

    The algorithm advances a sliding window of width *chunk_size* by
    ``chunk_size - overlap`` characters on each step, so consecutive
    chunks share *overlap* characters of context.

    Empty or whitespace-only strings return an empty list.

    Parameters
    ----------
    text : str
        The input text to split.
    chunk_size : int
        Maximum number of characters per chunk.
    overlap : int
        Number of characters shared between adjacent chunks.
        Must be strictly less than *chunk_size*.

    Returns
    -------
    List[str]
        Non-empty text chunks.

    Raises
    ------
    ValueError
        If ``overlap >= chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
        )

    text = text.strip()
    if not text:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def _process_single_pdf(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
) -> List[TextChunk]:
    """Parse one PDF and return all of its text chunks.

    Parameters
    ----------
    pdf_path : Path
        Absolute path to the PDF.
    chunk_size : int
        Target character length of each chunk.
    overlap : int
        Overlap in characters between adjacent chunks.

    Returns
    -------
    List[TextChunk]
        All chunks extracted from this PDF, ordered by page then position.
    """
    book_name = pdf_path.stem                  # filename without extension
    source = str(pdf_path)
    all_chunks: List[TextChunk] = []

    pages = _extract_pages(pdf_path)
    if not pages:
        logger.warning("'{}' yielded no extractable text — skipping.", pdf_path.name)
        return all_chunks

    global_chunk_idx = 0
    for page_number, page_text in pages:
        page_chunks = _split_into_chunks(page_text, chunk_size, overlap)
        for chunk_text in page_chunks:
            all_chunks.append(
                TextChunk(
                    text=chunk_text,
                    page_number=page_number,
                    source=source,
                    book_name=book_name,
                    chunk_index=global_chunk_idx,
                )
            )
            global_chunk_idx += 1

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_chunk_all_books(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[TextChunk]:
    """Discover and process every PDF in ``data/books/``.

    This is the primary entry point for the PDF processing pipeline.  It:

    1. Scans ``<project_root>/data/books/`` for ``.pdf`` files.
    2. Skips any file that is not a PDF.
    3. Extracts clean UTF-8 text from each page using PyMuPDF.
    4. Splits each page's text into overlapping character-level chunks.
    5. Returns all chunks across all books as a flat list.

    No file paths are accepted as input — book discovery is fully automatic.

    Parameters
    ----------
    chunk_size : int, optional
        Character length of each chunk.  Defaults to the value of the
        ``CHUNK_SIZE`` environment variable, or 500 if unset.
    overlap : int, optional
        Characters of overlap between adjacent chunks.  Defaults to the
        value of the ``CHUNK_OVERLAP`` environment variable, or 50 if
        unset.

    Returns
    -------
    List[TextChunk]
        Flat list of :class:`TextChunk` objects ordered by book, then
        page, then position within the page.  An empty list is returned
        if no processable PDFs are found.

    Raises
    ------
    FileNotFoundError
        If the ``data/books/`` directory does not exist at all.
    """
    logger.info("Scanning for PDF books in: {}", BOOKS_DIR)

    pdf_files = _discover_pdfs(BOOKS_DIR)

    if not pdf_files:
        logger.warning(
            "No PDF files found in {}.  "
            "Place your book PDFs there and restart.",
            BOOKS_DIR,
        )
        return []

    logger.info(
        "Found {} book(s): {}",
        len(pdf_files),
        [p.name for p in pdf_files],
    )

    all_chunks: List[TextChunk] = []

    for pdf_path in pdf_files:
        logger.info("Processing '{}' …", pdf_path.name)
        chunks = _process_single_pdf(pdf_path, chunk_size, overlap)
        logger.info(
            "  '{}' → {} chunk(s) extracted.",
            pdf_path.name,
            len(chunks),
        )
        all_chunks.extend(chunks)

    logger.info(
        "PDF processing complete. Total chunks across all books: {}",
        len(all_chunks),
    )
    return all_chunks


def get_book_names() -> List[str]:
    """Return the stem names of all PDF files in the books directory.

    Useful for checking which books are available without processing them.

    Returns
    -------
    List[str]
        Sorted list of book names (filename stems without ``.pdf``).
        Returns an empty list if the directory is missing or empty.
    """
    try:
        return [p.stem for p in _discover_pdfs(BOOKS_DIR)]
    except FileNotFoundError:
        return []
