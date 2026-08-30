"""
test_guardrail.py
=================
Unit and integration tests for the :mod:`app.services.guardrail` module.

Test strategy
-------------
* **Unit tests** (``TestGuardrailTrivialRejection``) — exercise the fast
  pre-filter path using a *mock* vector store so no real ChromaDB or
  embedding model is needed.
* **Integration tests** (``TestGuardrailWithMockStore``) — use a mock
  vector store whose ``similarity_search`` returns controlled scores,
  verifying the threshold logic end-to-end.
* **Parametrised smoke test** (``TestGuardrailCases``) — 10 labelled
  cases (5 allowed, 5 rejected) run against the mock store to document
  expected behaviour clearly.

Running the tests
-----------------
    # From backend/ with the virtual environment activated:
    pytest tests/unit/test_guardrail.py -v

    # With coverage:
    pytest tests/unit/test_guardrail.py -v --cov=app.services.guardrail
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ── Make sure the app package is importable ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.guardrail import (
    REJECTION_MESSAGE,
    SIMILARITY_THRESHOLD,
    Guardrail,
    GuardrailResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_store(
    best_score: float = 0.5,
    chunk_count: int = 10,
) -> MagicMock:
    """Return a mock VectorStore that returns a controlled similarity score.

    Parameters
    ----------
    best_score : float
        The cosine similarity score returned for the first (best) chunk.
    chunk_count : int
        The value returned by ``store.count()``.  Set to 0 to simulate an
        empty collection.

    Returns
    -------
    MagicMock
        A mock that satisfies the VectorStore interface expected by
        :class:`~app.services.guardrail.Guardrail`.
    """
    store = MagicMock()
    store.count.return_value = chunk_count

    # Produce one realistic chunk at ``best_score`` plus a lower-scored filler.
    chunk_meta = {
        "text": "Sample text about knowledge economy and globalisation.",
        "book_name": "knowledge_economy",
        "page_number": 3,
        "source": "/data/books/knowledge_economy.pdf",
        "chunk_index": 7,
    }
    filler_meta = {
        "text": "Another passage from the book.",
        "book_name": "knowledge_economy",
        "page_number": 5,
        "source": "/data/books/knowledge_economy.pdf",
        "chunk_index": 12,
    }
    store.similarity_search.return_value = [
        (chunk_meta, best_score),
        (filler_meta, max(0.0, best_score - 0.1)),
    ]
    return store


def _make_guardrail(best_score: float = 0.5, chunk_count: int = 10) -> Guardrail:
    """Convenience factory: guardrail backed by a mock store."""
    return Guardrail(_make_mock_store(best_score, chunk_count))


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 1 — GuardrailResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailResult:
    """Verify the GuardrailResult dataclass and its serialisation."""

    def test_to_dict_allowed(self):
        """to_dict() includes all public fields when allowed=True."""
        result = GuardrailResult(
            allowed=True,
            message="",
            chunks=[{"text": "hello"}],
            best_score=0.72,
        )
        d = result.to_dict()
        assert d["allowed"] is True
        assert d["message"] == ""
        assert len(d["chunks"]) == 1
        assert d["best_score"] == 0.72

    def test_to_dict_rejected(self):
        """to_dict() omits rejection_reason (internal field)."""
        result = GuardrailResult(
            allowed=False,
            message=REJECTION_MESSAGE,
            best_score=0.1,
            rejection_reason="Score too low",
        )
        d = result.to_dict()
        assert d["allowed"] is False
        assert d["message"] == REJECTION_MESSAGE
        assert "rejection_reason" not in d

    def test_chunks_default_empty(self):
        """chunks defaults to an empty list — no shared mutable default."""
        r1 = GuardrailResult(allowed=True)
        r2 = GuardrailResult(allowed=True)
        r1.chunks.append({"x": 1})
        assert r2.chunks == [], "Mutable default shared between instances"


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 2 — Trivial rejection (no vector store query)
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailTrivialRejection:
    """Questions blocked before the vector store is ever queried."""

    def setup_method(self):
        self.guard = _make_guardrail()

    def _assert_rejected_without_search(self, question: str):
        result = self.guard.check(question)
        assert result.allowed is False
        assert result.message == REJECTION_MESSAGE
        # Vector store should NOT have been queried.
        self.guard._store.similarity_search.assert_not_called()

    # ── Test cases ────────────────────────────────────────────────────

    def test_empty_string(self):
        """Empty input is rejected immediately."""
        result = self.guard.check("")
        assert result.allowed is False

    def test_whitespace_only(self):
        """Whitespace-only input is rejected immediately."""
        result = self.guard.check("     ")
        assert result.allowed is False

    def test_greeting_hi(self):
        """'hi' is too short and is a known greeting."""
        self._assert_rejected_without_search("hi")

    def test_greeting_hello(self):
        """'hello' is recognised as small talk."""
        self._assert_rejected_without_search("hello")

    def test_greeting_how_are_you(self):
        """'how are you' is recognised as small talk."""
        self._assert_rejected_without_search("how are you")

    def test_greeting_good_morning(self):
        """'good morning' is recognised as small talk."""
        self._assert_rejected_without_search("good morning")

    def test_single_character(self):
        """A single character is too short to carry semantic meaning."""
        self._assert_rejected_without_search("?")

    def test_thanks(self):
        """'thanks' is recognised as filler / small talk."""
        self._assert_rejected_without_search("thanks")


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 3 — Empty collection
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailEmptyCollection:
    """Behaviour when no books have been ingested yet."""

    def test_any_question_rejected_when_store_empty(self):
        """All substantive questions are rejected if the store is empty."""
        guard = _make_guardrail(chunk_count=0)
        result = guard.check("What is the knowledge economy?")
        assert result.allowed is False
        assert "No books" in result.message
        guard._store.similarity_search.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 4 — Threshold logic
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailThreshold:
    """Verify the boundary conditions around the similarity threshold."""

    def test_score_above_threshold_is_allowed(self):
        """A score clearly above the threshold allows the question through."""
        guard = _make_guardrail(best_score=0.75)
        result = guard.check("What does Talal say about globalisation?")
        assert result.allowed is True
        assert result.best_score == 0.75

    def test_score_at_threshold_is_allowed(self):
        """A score exactly equal to the threshold is allowed (inclusive lower bound)."""
        guard = _make_guardrail(best_score=SIMILARITY_THRESHOLD)
        result = guard.check("Tell me about Talal Abu Ghazaleh's vision.")
        assert result.allowed is True

    def test_score_just_below_threshold_is_rejected(self):
        """A score just below the threshold is rejected."""
        guard = _make_guardrail(best_score=SIMILARITY_THRESHOLD - 0.01)
        result = guard.check("Tell me about Talal Abu Ghazaleh's vision.")
        assert result.allowed is False
        assert result.message == REJECTION_MESSAGE

    def test_score_zero_is_rejected(self):
        """A zero similarity score is always rejected."""
        guard = _make_guardrail(best_score=0.0)
        result = guard.check("What is the stock market doing today?")
        assert result.allowed is False

    def test_custom_threshold_respected(self):
        """A custom threshold overrides the default."""
        store = _make_mock_store(best_score=0.45)
        guard = Guardrail(store, threshold=0.50)    # stricter than default
        result = guard.check("Tell me about the Arab world economy.")
        assert result.allowed is False              # 0.45 < 0.50

    def test_chunks_returned_on_allow(self):
        """When allowed, the chunks payload is non-empty and contains scores."""
        guard = _make_guardrail(best_score=0.68)
        result = guard.check("What is Talal's view on Arab education?")
        assert result.allowed is True
        assert len(result.chunks) > 0
        first_chunk = result.chunks[0]
        assert "text" in first_chunk
        assert "book_name" in first_chunk
        assert "score" in first_chunk
        assert first_chunk["score"] == pytest.approx(0.68, abs=0.01)

    def test_chunks_empty_on_rejection(self):
        """When rejected, no chunks are returned."""
        guard = _make_guardrail(best_score=0.10)
        result = guard.check("Who is the president of France?")
        assert result.allowed is False
        assert result.chunks == []


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 5 — 10 labelled parametrised cases
# ─────────────────────────────────────────────────────────────────────────────

# Each case: (label, question, mock_score, expected_allowed)
# Mock scores simulate what ChromaDB would return for that question type.
GUARDRAIL_CASES = [
    # ── ALLOWED (score ≥ 0.30) ───────────────────────────────────────────
    (
        "book_topic_knowledge_economy",
        "What does Talal Abu Ghazaleh say about the knowledge economy?",
        0.72,
        True,
    ),
    (
        "book_topic_globalisation",
        "How does globalisation affect Arab countries according to the books?",
        0.65,
        True,
    ),
    (
        "book_topic_education",
        "What is Talal's perspective on education and intellectual capital?",
        0.60,
        True,
    ),
    (
        "book_topic_arab_world",
        "What challenges does the Arab world face in the digital age?",
        0.55,
        True,
    ),
    (
        "book_topic_borderline_pass",
        "Can you summarise the main ideas from the books?",
        0.35,
        True,
    ),
    # ── REJECTED (score < 0.30) ──────────────────────────────────────────
    (
        "unrelated_weather",
        "What is the weather like in Amman today?",
        0.05,
        False,
    ),
    (
        "unrelated_sports",
        "Who won the Champions League last season?",
        0.04,
        False,
    ),
    (
        "unrelated_cooking",
        "How do I make mansaf at home?",
        0.08,
        False,
    ),
    (
        "unrelated_other_person",
        "Tell me about Elon Musk's business strategy.",
        0.12,
        False,
    ),
    (
        "borderline_fail",
        "What is the best investment strategy for 2025?",
        0.22,
        False,
    ),
]


class TestGuardrailCases:
    """10 parametrised test cases covering allowed and rejected scenarios."""

    @pytest.mark.parametrize(
        "label, question, mock_score, expected_allowed",
        GUARDRAIL_CASES,
        ids=[c[0] for c in GUARDRAIL_CASES],
    )
    def test_case(
        self,
        label: str,
        question: str,
        mock_score: float,
        expected_allowed: bool,
    ):
        """Each labelled case produces the expected allowed/rejected outcome.

        Parameters
        ----------
        label : str
            Human-readable test identifier (shown in pytest output).
        question : str
            The simulated user question.
        mock_score : float
            The similarity score the mock vector store returns.
        expected_allowed : bool
            Whether the guardrail should allow (``True``) or block
            (``False``) the question.
        """
        guard = _make_guardrail(best_score=mock_score)
        result = guard.check(question)

        assert result.allowed is expected_allowed, (
            f"[{label}] Question: '{question}'\n"
            f"  score={mock_score}, threshold={SIMILARITY_THRESHOLD}\n"
            f"  expected allowed={expected_allowed}, got allowed={result.allowed}\n"
            f"  rejection_reason={result.rejection_reason}"
        )

        if expected_allowed:
            assert result.message == "", (
                f"[{label}] Allowed result should have empty message, "
                f"got: '{result.message}'"
            )
            assert len(result.chunks) > 0, (
                f"[{label}] Allowed result should contain chunks"
            )
        else:
            assert result.message == REJECTION_MESSAGE, (
                f"[{label}] Rejected result should have the standard message"
            )
            assert result.chunks == [], (
                f"[{label}] Rejected result should have no chunks"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test suite 6 — Robustness / edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestGuardrailRobustness:
    """Edge cases that should not raise exceptions."""

    def test_very_long_question(self):
        """A very long question is handled without error."""
        long_q = "What does Talal say about " + "globalisation " * 200
        guard = _make_guardrail(best_score=0.55)
        result = guard.check(long_q)
        assert isinstance(result, GuardrailResult)

    def test_question_with_special_characters(self):
        """Questions containing special characters are handled gracefully."""
        guard = _make_guardrail(best_score=0.50)
        result = guard.check("What is Talal's view on 'AI' & the economy?!")
        assert isinstance(result, GuardrailResult)

    def test_search_exception_returns_rejected(self):
        """If similarity_search raises, the guardrail returns a safe rejection."""
        store = _make_mock_store()
        store.similarity_search.side_effect = RuntimeError("DB connection failed")
        guard = Guardrail(store)
        result = guard.check("What does Talal say about education?")
        assert result.allowed is False
        assert result.rejection_reason is not None
        assert "Search error" in result.rejection_reason

    def test_threshold_property(self):
        """The threshold property reflects the value set at construction."""
        guard = Guardrail(_make_mock_store(), threshold=0.45)
        assert guard.threshold == 0.45


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for running directly with `python test_guardrail.py`
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
