"""
vector_store.py
===============
Manages the local ChromaDB vector store used by the RAG pipeline.

Responsibilities
----------------
* Initialise (or reopen) a persistent Chroma collection on disk.
* Embed text chunks with ``sentence-transformers/all-MiniLM-L6-v2``.
* Persist chunks with their metadata (book name, page, source).
* Avoid re-embedding books that are already stored in the collection.
* Provide a ``similarity_search`` function that returns the most
  relevant chunks **and** their cosine similarity scores.

Typical usage
-------------
    from app.db.vector_store import VectorStore

    store = VectorStore()
    store.ingest_chunks(chunks)          # chunks from pdf_processor
    results = store.similarity_search("What is knowledge economy?", top_k=4)
    for chunk, score in results:
        print(f"[{score:.3f}] {chunk['book_name']} p.{chunk['page_number']}")
        print(chunk["text"][:120])
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.utils.pdf_processor import TextChunk

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DB_DIR: Path = Path(
    os.getenv("CHROMA_DB_DIR", str(_PROJECT_ROOT / "data" / "chroma_db"))
)
COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "talal_books")
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore class
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """Wrapper around a persistent ChromaDB collection with sentence-transformer
    embeddings.

    The collection is stored on disk at ``CHROMA_DB_DIR`` and survives
    application restarts.  Books that are already embedded are detected by
    checking the ``book_name`` metadata field in the collection, so they are
    never re-processed.

    Parameters
    ----------
    chroma_dir : Path, optional
        Override the default ChromaDB persistence directory.
    collection_name : str, optional
        Override the default collection name.
    embedding_model : str, optional
        Override the default sentence-transformer model identifier.
    device : str, optional
        PyTorch device for the embedding model (``"cpu"`` or ``"cuda"``).

    Attributes
    ----------
    model : SentenceTransformer
        The loaded embedding model (lazy-initialised on first use).
    collection : chromadb.Collection
        The active ChromaDB collection.
    """

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
    ) -> None:
        self._chroma_dir = chroma_dir
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model
        self._device = device

        self._model: SentenceTransformer | None = None
        self._collection: chromadb.Collection | None = None

        self._init_collection()

    # ──────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────

    def _init_collection(self) -> None:
        """Create or reopen the persistent ChromaDB collection.

        The database is persisted to ``self._chroma_dir``.  If the
        collection already exists its data is preserved; documents are
        never wiped on startup.
        """
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialising ChromaDB at: {}", self._chroma_dir)

        client = chromadb.PersistentClient(
            path=str(self._chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},   # use cosine distance
        )

        doc_count = self._collection.count()
        logger.info(
            "Collection '{}' ready — {} document(s) already stored.",
            self._collection_name,
            doc_count,
        )

    def _get_model(self) -> SentenceTransformer:
        """Lazily load and return the sentence-transformer model.

        The model is downloaded from HuggingFace Hub on the first call
        and cached locally by the ``sentence-transformers`` library.

        Returns
        -------
        SentenceTransformer
            Loaded embedding model.
        """
        if self._model is None:
            logger.info(
                "Loading embedding model '{}' on device '{}'…",
                self._embedding_model_name,
                self._device,
            )
            self._model = SentenceTransformer(
                self._embedding_model_name,
                device=self._device,
            )
            logger.info("Embedding model loaded.")
        return self._model

    # ──────────────────────────────────────────
    # Book tracking
    # ──────────────────────────────────────────

    def get_indexed_books(self) -> List[str]:
        """Return the names of books that have already been embedded.

        This is used to skip re-processing books on application restart.

        Returns
        -------
        List[str]
            Unique ``book_name`` values found in the collection's metadata.
            Returns an empty list if the collection is empty.
        """
        if self._collection.count() == 0:
            return []

        # Retrieve all metadata without embedding vectors (efficient).
        result = self._collection.get(include=["metadatas"])
        book_names = {
            meta["book_name"]
            for meta in result["metadatas"]
            if meta and "book_name" in meta
        }
        return sorted(book_names)

    def is_book_indexed(self, book_name: str) -> bool:
        """Check whether a specific book is already in the collection.

        Parameters
        ----------
        book_name : str
            The ``book_name`` value to look for (PDF stem, e.g.
            ``"knowledge_economy"``).

        Returns
        -------
        bool
            ``True`` if at least one chunk from this book is stored.
        """
        result = self._collection.get(
            where={"book_name": book_name},
            limit=1,
            include=["metadatas"],
        )
        return len(result["ids"]) > 0

    # ──────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────

    def ingest_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 64,
    ) -> int:
        """Embed and store a list of :class:`~app.utils.pdf_processor.TextChunk`
        objects.

        Books that are already present in the collection are skipped
        entirely — no duplicate embeddings are created.

        Parameters
        ----------
        chunks : List[TextChunk]
            Chunks produced by :func:`~app.utils.pdf_processor.load_and_chunk_all_books`.
        batch_size : int, optional
            Number of chunks to embed and upsert per batch.  Smaller
            batches use less memory; larger batches are faster.  Default
            is 64.

        Returns
        -------
        int
            Number of new chunks actually stored (0 if all books were
            already indexed).
        """
        if not chunks:
            logger.warning("ingest_chunks called with an empty list — nothing to do.")
            return 0

        # ── Filter out already-indexed books ──────────────────────────
        indexed_books = set(self.get_indexed_books())
        new_chunks = [c for c in chunks if c.book_name not in indexed_books]

        skipped_books = {c.book_name for c in chunks} & indexed_books
        if skipped_books:
            logger.info(
                "Skipping {} already-indexed book(s): {}",
                len(skipped_books),
                sorted(skipped_books),
            )

        if not new_chunks:
            logger.info("All books are already indexed — no ingestion needed.")
            return 0

        new_book_names = sorted({c.book_name for c in new_chunks})
        logger.info(
            "Ingesting {} chunk(s) from {} new book(s): {}",
            len(new_chunks),
            len(new_book_names),
            new_book_names,
        )

        model = self._get_model()
        total_stored = 0

        # ── Batch embed + upsert ──────────────────────────────────────
        for batch_start in range(0, len(new_chunks), batch_size):
            batch = new_chunks[batch_start : batch_start + batch_size]
            texts = [c.text for c in batch]

            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            self._collection.upsert(
                ids=[str(uuid.uuid4()) for _ in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "book_name": c.book_name,
                        "page_number": c.page_number,
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                    }
                    for c in batch
                ],
            )
            total_stored += len(batch)
            logger.debug(
                "  Stored batch {}/{} ({} chunks)",
                batch_start // batch_size + 1,
                -(-len(new_chunks) // batch_size),   # ceiling division
                len(batch),
            )

        logger.info(
            "Ingestion complete — {} new chunk(s) stored.  "
            "Collection total: {}.",
            total_stored,
            self._collection.count(),
        )
        return total_stored

    # ──────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[dict, float]]:
        """Find the most semantically similar chunks for a given query.

        The query is embedded with the same model used during ingestion,
        then ChromaDB's HNSW index is queried for the nearest neighbours
        using cosine distance.  Distances are converted to similarity
        scores so that **1.0 = perfect match** and **0.0 = no relation**.

        Parameters
        ----------
        query : str
            The user's natural-language question.
        top_k : int, optional
            Maximum number of results to return.  Defaults to 5.

        Returns
        -------
        List[Tuple[dict, float]]
            List of ``(metadata_dict, similarity_score)`` tuples sorted
            from most to least similar.  Each ``metadata_dict`` contains:

            * ``text``        – raw chunk text
            * ``book_name``   – source book identifier
            * ``page_number`` – 1-based page number
            * ``source``      – absolute path to the PDF
            * ``chunk_index`` – sequential index of the chunk

        Raises
        ------
        RuntimeError
            If the collection is empty (nothing has been ingested yet).
        """
        if self._collection.count() == 0:
            raise RuntimeError(
                "The vector store is empty.  "
                "Run the ingestion pipeline before querying."
            )

        model = self._get_model()
        query_embedding = model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns lists-of-lists (one list per query vector).
        documents: List[str] = result["documents"][0]
        metadatas: List[dict] = result["metadatas"][0]
        distances: List[float] = result["distances"][0]

        output: List[Tuple[dict, float]] = []
        for doc_text, meta, distance in zip(documents, metadatas, distances):
            # Cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1].
            # For well-normalised vectors distance ∈ [0, 1], so this gives
            # similarity ∈ [0, 1].
            similarity_score = float(1.0 - distance)
            enriched_meta = {
                "text": doc_text,
                "book_name": meta.get("book_name", "unknown"),
                "page_number": meta.get("page_number", -1),
                "source": meta.get("source", ""),
                "chunk_index": meta.get("chunk_index", -1),
            }
            output.append((enriched_meta, similarity_score))

        logger.debug(
            "similarity_search('{}…') → {} result(s), "
            "top score: {:.3f}",
            query[:50],
            len(output),
            output[0][1] if output else 0.0,
        )
        return output

    # ──────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────

    @property
    def collection(self) -> chromadb.Collection:
        """The underlying ChromaDB collection object."""
        return self._collection

    def count(self) -> int:
        """Return the total number of stored chunks.

        Returns
        -------
        int
            Document count in the collection.
        """
        return self._collection.count()
