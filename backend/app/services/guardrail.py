"""
guardrail.py
============
Topic-filtering guardrail for the Talal Abu Ghazaleh RAG assistant.

Before any question reaches the LLM, it is checked against the embedded
book content in ChromaDB.  If the question is semantically too distant
from the books (cosine similarity below ``SIMILARITY_THRESHOLD``), it is
rejected with a polite refusal message.

This approach is preferable to keyword blacklisting because it is
content-aware: a question like "What did he say about globalisation?"
passes even without the author's name, while "Who won the World Cup?"
is rejected even if it contains words that appear in the books.

Typical usage
-------------
    from app.services.guardrail import Guardrail, GuardrailResult

    guard = Guardrail(vector_store)
    result = guard.check("What is the knowledge economy?")

    if result.allowed:
        # Pass result.chunks directly to the RAG pipeline
        response = pipeline.answer_with_chunks(question, result.chunks)
    else:
        return result.message
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from app.db.vector_store import VectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD: float = float(
    os.getenv("TOPIC_SIMILARITY_THRESHOLD", 0.30)
)

RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", 4))

REJECTION_MESSAGE: str = (
    "I can only answer questions related to Talal Abu Ghazaleh's books."
)

# Short inputs (greetings, filler words) are often too brief to produce
# meaningful embeddings.  Any question shorter than this character count
# is rejected immediately without querying the vector store.
MIN_QUESTION_LENGTH: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """The outcome of a guardrail check.

    Attributes
    ----------
    allowed : bool
        ``True`` if the question is sufficiently related to the books and
        may proceed to the RAG pipeline.  ``False`` if it should be
        rejected.
    message : str
        A user-facing message.  When ``allowed`` is ``False`` this is the
        rejection string.  When ``allowed`` is ``True`` this is an empty
        string (the pipeline will produce its own answer).
    chunks : List[dict]
        The top-K retrieved chunks (with metadata and text) that passed
        the threshold.  Empty when ``allowed`` is ``False``.
        Each dict contains: ``text``, ``book_name``, ``page_number``,
        ``source``, ``chunk_index``, and ``score``.
    best_score : float
        The highest cosine similarity score found.  Useful for logging
        and debugging.  ``0.0`` when the collection is empty or the
        question was rejected before querying.
    rejection_reason : Optional[str]
        Human-readable reason for rejection (internal, not shown to the
        user).  ``None`` when the question is allowed.
    """

    allowed: bool
    message: str = ""
    chunks: List[dict] = field(default_factory=list)
    best_score: float = 0.0
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary suitable for a JSON response.

        Returns
        -------
        dict
            Keys: ``allowed``, ``message``, ``chunks``, ``best_score``.
            ``rejection_reason`` is omitted (internal use only).
        """
        return {
            "allowed": self.allowed,
            "message": self.message,
            "chunks": self.chunks,
            "best_score": round(self.best_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail class
# ─────────────────────────────────────────────────────────────────────────────

class Guardrail:
    """Semantic topic filter that gates access to the RAG pipeline.

    The guardrail embeds the incoming question and compares it against
    the book chunks already stored in ChromaDB.  Only questions whose
    best cosine similarity score meets or exceeds ``threshold`` are
    allowed through.

    Parameters
    ----------
    vector_store : VectorStore
        An initialised and populated :class:`~app.db.vector_store.VectorStore`.
    threshold : float, optional
        Minimum cosine similarity score required to allow a question.
        Defaults to the value of ``TOPIC_SIMILARITY_THRESHOLD`` env var,
        or ``0.30`` if unset.
    top_k : int, optional
        Number of chunks to retrieve for the similarity check (and to
        pass downstream to the RAG pipeline if allowed).  Defaults to
        ``RETRIEVER_TOP_K``.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        threshold: float = SIMILARITY_THRESHOLD,
        top_k: int = RETRIEVER_TOP_K,
    ) -> None:
        self._store = vector_store
        self._threshold = threshold
        self._top_k = top_k

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _is_trivially_rejectable(self, question: str) -> Tuple[bool, str]:
        """Detect questions that can be rejected without querying ChromaDB.

        Handles:
        * Empty or whitespace-only strings.
        * Very short inputs unlikely to carry semantic content (greetings,
          single words, filler phrases such as "hi", "ok", "thanks").

        Parameters
        ----------
        question : str
            Raw input from the user (not yet stripped).

        Returns
        -------
        Tuple[bool, str]
            ``(should_reject, reason)`` — if ``should_reject`` is ``True``
            the caller should return a rejection immediately.
        """
        stripped = question.strip()

        if not stripped:
            return True, "Empty input"

        if len(stripped) < MIN_QUESTION_LENGTH:
            return True, f"Input too short ({len(stripped)} chars) — likely a greeting or filler"

        # Common small-talk patterns (case-insensitive prefix match).
        small_talk_prefixes = (
            "hi", "hello", "hey", "hiya",
            "how are you", "what's up", "wassup",
            "good morning", "good afternoon", "good evening", "good night",
            "thanks", "thank you", "cheers", "bye", "goodbye", "see you",
            "lol", "haha", "ok", "okay", "sure", "nice", "cool",
        )
        lower = stripped.lower()
        for phrase in small_talk_prefixes:
            if lower == phrase or lower.startswith(phrase + " ") or lower.startswith(phrase + "!") or lower.startswith(phrase + ","):
                return True, f"Small talk / greeting detected: '{stripped}'"

        return False, ""

    def _build_chunks_payload(
        self,
        results: List[Tuple[dict, float]],
    ) -> List[dict]:
        """Merge similarity scores into the metadata dicts for downstream use.

        Parameters
        ----------
        results : List[Tuple[dict, float]]
            Output of :meth:`~app.db.vector_store.VectorStore.similarity_search`.

        Returns
        -------
        List[dict]
            Each dict contains all metadata fields plus a ``score`` key.
        """
        chunks = []
        for meta, score in results:
            chunk = {**meta, "score": round(score, 4)}
            chunks.append(chunk)
        return chunks

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def check(self, question: str) -> GuardrailResult:
        """Decide whether a question is related enough to the books to proceed.

        Algorithm
        ---------
        1. Trivial rejection — empty inputs and small talk are blocked
           immediately without touching the vector store.
        2. Vector store emptiness check — if no books have been ingested
           yet, all questions are rejected with an informative message.
        3. Similarity search — the question is embedded and compared
           against all stored chunks.
        4. Threshold check — if the best score is below ``self._threshold``
           the question is rejected; otherwise the matched chunks are
           returned for the RAG pipeline to consume.

        Parameters
        ----------
        question : str
            Raw user question (any length, any language).

        Returns
        -------
        GuardrailResult
            See :class:`GuardrailResult` for field descriptions.
            The ``chunks`` field is only populated when ``allowed`` is
            ``True``; the RAG pipeline should use these chunks directly
            instead of running a second similarity search.
        """
        logger.debug("Guardrail check for: '{}'", question[:80])

        # ── Step 1: Trivial rejection ─────────────────────────────────
        should_reject, reason = self._is_trivially_rejectable(question)
        if should_reject:
            logger.info("Guardrail BLOCKED (trivial): {} | reason: {}", repr(question[:60]), reason)
            return GuardrailResult(
                allowed=False,
                message=REJECTION_MESSAGE,
                best_score=0.0,
                rejection_reason=reason,
            )

        # ── Step 2: Empty collection check ───────────────────────────
        if self._store.count() == 0:
            reason = "Vector store is empty — no books have been ingested yet"
            logger.warning("Guardrail BLOCKED: {}", reason)
            return GuardrailResult(
                allowed=False,
                message="No books have been loaded yet. Please contact the administrator.",
                best_score=0.0,
                rejection_reason=reason,
            )

        # ── Step 3: Similarity search ─────────────────────────────────
        try:
            results = self._store.similarity_search(question, top_k=self._top_k)
        except Exception as exc:
            logger.error("Guardrail similarity search failed: {}", exc)
            return GuardrailResult(
                allowed=False,
                message="An internal error occurred. Please try again.",
                best_score=0.0,
                rejection_reason=f"Search error: {exc}",
            )

        best_score = results[0][1] if results else 0.0

        # ── Step 4: Threshold decision ────────────────────────────────
        if best_score < self._threshold:
            reason = (
                f"Best similarity score {best_score:.4f} is below "
                f"threshold {self._threshold}"
            )
            logger.info(
                "Guardrail BLOCKED: '{}…' | score={:.4f} < threshold={:.2f}",
                question[:50],
                best_score,
                self._threshold,
            )
            return GuardrailResult(
                allowed=False,
                message=REJECTION_MESSAGE,
                best_score=best_score,
                rejection_reason=reason,
            )

        # ── Allowed ───────────────────────────────────────────────────
        chunks = self._build_chunks_payload(results)
        logger.info(
            "Guardrail ALLOWED: '{}…' | score={:.4f} | {} chunk(s) returned",
            question[:50],
            best_score,
            len(chunks),
        )
        return GuardrailResult(
            allowed=True,
            message="",
            chunks=chunks,
            best_score=best_score,
        )

    @property
    def threshold(self) -> float:
        """The current similarity threshold (read-only)."""
        return self._threshold
