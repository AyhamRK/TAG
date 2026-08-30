"""
ingest.py
=========
Standalone script that scans ``data/books/``, extracts text from all PDFs,
embeds the chunks, and stores them in the local ChromaDB vector store.

Run this once after adding new books, or whenever you want to refresh the
index.  Books that are already embedded are skipped automatically — safe
to run multiple times.

Usage
-----
    # From the backend/ directory with the virtual environment activated:
    python ingest.py

    # With a custom books directory (overrides .env):
    PDF_DATA_DIR=./my_custom_folder python ingest.py
"""

import sys
from pathlib import Path

from loguru import logger

# Ensure the app package is importable when running from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.utils.pdf_processor import load_and_chunk_all_books
from app.db.vector_store import VectorStore


def run_ingestion() -> None:
    """Execute the full ingestion pipeline.

    1. Discover and parse all PDFs in ``data/books/``.
    2. Open (or create) the ChromaDB collection.
    3. Embed and store any books not yet in the collection.
    4. Log a summary of what was processed.
    """
    logger.info("═══════════════════════════════════════")
    logger.info(" Talal Abu Ghazaleh RAG — Book Ingestion")
    logger.info("═══════════════════════════════════════")

    # ── Step 1: Parse PDFs ────────────────────────────────────────────
    logger.info("Step 1/3 — Loading and chunking PDFs…")
    try:
        chunks = load_and_chunk_all_books()
    except FileNotFoundError as exc:
        logger.error("{}", exc)
        sys.exit(1)

    if not chunks:
        logger.warning("No chunks were produced.  Add PDF books to data/books/ and retry.")
        sys.exit(0)

    logger.info("Total chunks loaded: {}", len(chunks))

    # ── Step 2: Initialise vector store ──────────────────────────────
    logger.info("Step 2/3 — Initialising vector store…")
    store = VectorStore()
    already_indexed = store.get_indexed_books()
    if already_indexed:
        logger.info("Already indexed books: {}", already_indexed)

    # ── Step 3: Embed and store ───────────────────────────────────────
    logger.info("Step 3/3 — Embedding and storing new chunks…")
    stored = store.ingest_chunks(chunks)

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("═══════════════════════════════════════")
    if stored == 0:
        logger.info("Nothing new to store — all books are already indexed.")
    else:
        logger.info("Ingestion complete.  {} new chunk(s) stored.", stored)
    logger.info("Total chunks in collection: {}", store.count())
    logger.info("Indexed books: {}", store.get_indexed_books())
    logger.info("═══════════════════════════════════════")


if __name__ == "__main__":
    run_ingestion()
