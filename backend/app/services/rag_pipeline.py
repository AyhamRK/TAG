"""
rag_pipeline.py
===============
Orchestrates the full Retrieval-Augmented Generation (RAG) cycle:
retrieve → detect intent → generate with the right model → return.

Two models are used locally — no external API calls:

* **Flan-T5-Large** (``google/flan-t5-large``) — factual Q&A and
  key-point listing.  Fast, small, accurate for direct questions.
* **BART-Large-CNN** (``facebook/bart-large-cnn``) — summarisation.
  Purpose-built for abstractive summarisation; produces coherent,
  multi-sentence summaries from retrieved passages.

Both models are loaded **lazily** on first use and kept in memory for
the lifetime of the process.

Typical usage
-------------
    from app.services.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(vector_store)
    response = pipeline.answer("Summarize the book because i like the truth")
    print(response.answer)
    for src in response.sources:
        print(src.book_name, src.page_number, src.score)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
)

from app.db.vector_store import VectorStore

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "google/flan-t5-large")
SUMMARISER_MODEL_NAME: str = os.getenv(
    "SUMMARISER_MODEL_NAME", "facebook/bart-large-cnn"
)
LLM_DEVICE: str = os.getenv("LLM_DEVICE", "cpu")
LLM_MAX_NEW_TOKENS: int = int(os.getenv("LLM_MAX_NEW_TOKENS", 512))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.1))
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", 8))

MIN_SIMILARITY_THRESHOLD: float = float(
    os.getenv("TOPIC_SIMILARITY_THRESHOLD", 0.25)
)

# Flan-T5 context budget (~512 tokens * 4 chars/token)
QA_MAX_CONTEXT_CHARS: int = 6_000

# BART can handle longer input (1024 tokens)
SUMMARISER_MAX_CONTEXT_CHARS: int = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    """A single source chunk cited in the generated answer.

    Attributes
    ----------
    book_name : str
        PDF filename stem (e.g. ``"because_i_like_the_truth"``).
    page_number : int
        1-based page number within the source PDF.
    source : str
        Absolute path to the PDF file on disk.
    score : float
        Cosine similarity ∈ [0, 1] between the query and this chunk.
    text_snippet : str
        First 200 characters of the chunk for reference.
    """

    book_name: str
    page_number: int
    source: str
    score: float
    text_snippet: str


@dataclass
class RAGResponse:
    """Complete output of one RAG pipeline call.

    Attributes
    ----------
    answer : str
        The model-generated answer grounded in retrieved context.
    sources : List[SourceReference]
        Chunks used to generate the answer, ordered by relevance.
    query : str
        The original user question.
    context_used : bool
        ``True`` if at least one chunk met the similarity threshold.
    model_used : str
        Which model produced the answer (``"flan-t5"`` or ``"bart"``).
    """

    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    query: str = ""
    context_used: bool = True
    model_used: str = "flan-t5"


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """Dual-model RAG pipeline.

    Routes each question to the appropriate model based on detected
    intent:

    * Summarisation questions → BART-Large-CNN
    * Q&A and listing questions → Flan-T5-Large

    Both models load lazily on first use.

    Parameters
    ----------
    vector_store : VectorStore
        Populated vector store instance.
    model_name : str, optional
        Flan-T5 HuggingFace identifier.
    summariser_model_name : str, optional
        BART HuggingFace identifier.
    device : str, optional
        PyTorch device (``"cpu"`` or ``"cuda"``).
    max_new_tokens : int, optional
        Max tokens the models may generate.
    top_k : int, optional
        Chunks retrieved per query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = LLM_MODEL_NAME,
        summariser_model_name: str = SUMMARISER_MODEL_NAME,
        device: str = LLM_DEVICE,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        top_k: int = RETRIEVER_TOP_K,
    ) -> None:
        self._vector_store = vector_store
        self._model_name = model_name
        self._summariser_model_name = summariser_model_name
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._top_k = top_k

        # Flan-T5
        self._t5_tokenizer: AutoTokenizer | None = None
        self._t5_model: T5ForConditionalGeneration | None = None

        # BART
        self._bart_tokenizer: AutoTokenizer | None = None
        self._bart_model: BartForConditionalGeneration | None = None

    # ──────────────────────────────────────────
    # Model loading (lazy)
    # ──────────────────────────────────────────

    def _load_t5(self) -> None:
        """Lazily load Flan-T5-Large for Q&A and listing.

        Downloaded once from HuggingFace and cached in
        ``~/.cache/huggingface/``.  Subsequent calls are instant.
        """
        if self._t5_model is not None:
            return
        logger.info(
            "Loading Flan-T5 '{}' on '{}'…", self._model_name, self._device
        )
        self._t5_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._t5_model = T5ForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=torch.float32,
        ).to(self._device)
        self._t5_model.eval()
        logger.info("Flan-T5 ready.")

    def _load_bart(self) -> None:
        """Lazily load BART-Large-CNN for summarisation.

        BART-Large-CNN (~1.6 GB) is a news summarisation model that
        produces fluent, abstractive multi-sentence summaries.  It is
        only loaded when a summarisation question is detected, keeping
        startup fast.
        """
        if self._bart_model is not None:
            return
        logger.info(
            "Loading BART summariser '{}' on '{}'…",
            self._summariser_model_name,
            self._device,
        )
        self._bart_tokenizer = AutoTokenizer.from_pretrained(
            self._summariser_model_name
        )
        self._bart_model = BartForConditionalGeneration.from_pretrained(
            self._summariser_model_name,
            torch_dtype=torch.float32,
        ).to(self._device)
        self._bart_model.eval()
        logger.info("BART summariser ready.")

    # ──────────────────────────────────────────
    # Intent detection
    # ──────────────────────────────────────────

    @staticmethod
    def _detect_intent(question: str) -> str:
        """Classify a question into one of three intent buckets.

        Intent determines which model and prompt template are used:

        * ``"summarise"`` → BART-Large-CNN with summarisation prompt.
        * ``"list"``      → Flan-T5 with bullet-point prompt.
        * ``"qa"``        → Flan-T5 with strict Q&A prompt (default).

        Parameters
        ----------
        question : str
            Raw user question (case-insensitive matching applied).

        Returns
        -------
        str
            One of ``"summarise"``, ``"list"``, or ``"qa"``.
        """
        lower = question.lower()

        summarise_triggers = (
            "summarize", "summarise", "summary", "summarization",
            "what is this book about", "what is the book about",
            "give me an overview", "overview of", "brief overview",
            "what does the book cover", "describe the book",
            "what topics does", "main theme", "main themes",
            "tell me about the book",
        )
        list_triggers = (
            "list", "what are the", "give me the", "enumerate",
            "what were the", "examples of", "key points",
            "main points", "key ideas", "key lessons",
            "achievements", "contributions",
        )

        if any(t in lower for t in summarise_triggers):
            return "summarise"
        if any(t in lower for t in list_triggers):
            return "list"
        return "qa"

    # ──────────────────────────────────────────
    # Prompt construction (Flan-T5 only)
    # ──────────────────────────────────────────

    @staticmethod
    def _build_t5_prompt(question: str, context_passages: List[str]) -> str:
        """Build the Flan-T5 prompt for Q&A or listing intents.

        Instructs the model to answer **only** from the provided passages.
        For listing intent, requests bullet-point output.
        For Q&A intent, requests a direct factual answer.

        Parameters
        ----------
        question : str
            The user's question.
        context_passages : List[str]
            Retrieved chunks ordered by relevance.

        Returns
        -------
        str
            Complete prompt string ready for tokenisation.
        """
        if not context_passages:
            return (
                "You are a strict question-answering assistant. "
                "No relevant passages were found in the books. "
                "Reply with: 'I could not find an answer to this question "
                "in the available books.'\n\n"
                f"Question: {question}\nAnswer:"
            )

        numbered = "\n\n".join(
            f"[Passage {i + 1}]\n{p}" for i, p in enumerate(context_passages)
        )
        intent = RAGPipeline._detect_intent(question)

        if intent == "list":
            return (
                "You are a helpful assistant. "
                "Using ONLY the passages below, list the key points "
                "relevant to the request. "
                "Present each point on a new line starting with a dash. "
                "Do NOT add information not in the passages.\n\n"
                f"Passages:\n{numbered}\n\n"
                f"Request: {question}\nKey points:"
            )

        # Default Q&A
        return (
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the passages provided. "
            "Do NOT use your training knowledge. "
            "If the answer is not in the passages reply with: "
            "'I could not find an answer to this question in the "
            "available books.'\n\n"
            f"Context passages:\n{numbered}\n\n"
            f"Question: {question}\nAnswer:"
        )

    # ──────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────

    def _generate_with_t5(self, prompt: str) -> str:
        """Run Flan-T5-Large on a prompt and return the decoded text.

        Parameters
        ----------
        prompt : str
            Full prompt string from :meth:`_build_t5_prompt`.

        Returns
        -------
        str
            Generated text with leading/trailing whitespace stripped.
        """
        self._load_t5()
        inputs = self._t5_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._t5_model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=LLM_TEMPERATURE,
                do_sample=LLM_TEMPERATURE > 0,
                num_beams=4 if LLM_TEMPERATURE == 0 else 1,
                early_stopping=True,
            )
        return self._t5_tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

    def _generate_with_bart(self, context_text: str) -> str:
        """Run BART-Large-CNN on concatenated context and return a summary.

        BART is fed the raw context text directly (no instruction
        wrapping needed — it was fine-tuned for summarisation).  The
        ``min_length`` parameter ensures the output is at least a
        paragraph rather than a single sentence.

        Parameters
        ----------
        context_text : str
            Concatenated retrieved passages (plain text, no labels).

        Returns
        -------
        str
            Abstractive summary produced by BART.
        """
        self._load_bart()
        inputs = self._bart_tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._bart_model.generate(
                **inputs,
                max_length=300,
                min_length=80,        # ensure a proper paragraph
                num_beams=4,
                length_penalty=2.0,   # encourage longer output
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self._bart_tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def answer(self, question: str) -> RAGResponse:
        """Answer a question by routing to the appropriate model.

        Routing logic
        -------------
        1. Retrieve top-K chunks from the vector store.
        2. Filter chunks below the minimum similarity threshold.
        3. Detect intent (``summarise`` / ``list`` / ``qa``).
        4a. **Summarise** → concatenate passages, run BART-Large-CNN.
        4b. **List / Q&A** → build instruction prompt, run Flan-T5-Large.
        5. Return answer + source citations.

        Parameters
        ----------
        question : str
            The user's natural-language question.

        Returns
        -------
        RAGResponse
            Answer text, source references, original query, context flag,
            and the name of the model used.

        Raises
        ------
        RuntimeError
            If the vector store is empty.
        """
        logger.info("RAG query: '{}'", question)

        # ── Step 1: Retrieve ──────────────────────────────────────────
        raw_results = self._vector_store.similarity_search(
            question, top_k=self._top_k
        )

        # ── Step 2: Filter by threshold ───────────────────────────────
        filtered = [
            (meta, score)
            for meta, score in raw_results
            if score >= MIN_SIMILARITY_THRESHOLD
        ]

        if not filtered:
            logger.warning("No chunks above threshold for: '{}'", question)

        context_used = len(filtered) > 0

        # ── Step 3: Detect intent ─────────────────────────────────────
        intent = self._detect_intent(question)
        logger.debug("Detected intent: '{}'", intent)

        # ── Step 4: Build context and generate ────────────────────────
        sources: List[SourceReference] = []

        if intent == "summarise":
            # BART path — feed more context, no instruction wrapping
            max_chars = SUMMARISER_MAX_CONTEXT_CHARS
            context_parts: List[str] = []
            total = 0
            for meta, score in filtered:
                text = meta["text"]
                if total + len(text) > max_chars:
                    remaining = max_chars - total
                    if remaining > 100:
                        context_parts.append(text[:remaining])
                    break
                context_parts.append(text)
                total += len(text)
                sources.append(
                    SourceReference(
                        book_name=meta["book_name"],
                        page_number=meta["page_number"],
                        source=meta["source"],
                        score=round(score, 4),
                        text_snippet=text[:200],
                    )
                )

            if not context_parts:
                return RAGResponse(
                    answer="I could not find an answer to this question in the available books.",
                    sources=[],
                    query=question,
                    context_used=False,
                    model_used="bart",
                )

            context_text = " ".join(context_parts)
            logger.debug(
                "BART summarising {} chars from {} chunk(s).",
                len(context_text),
                len(context_parts),
            )
            answer_text = self._generate_with_bart(context_text)
            model_used = "bart"

        else:
            # Flan-T5 path — instruction prompt with labelled passages
            max_chars = QA_MAX_CONTEXT_CHARS
            passages: List[str] = []
            total = 0
            for meta, score in filtered:
                text = meta["text"]
                if total + len(text) > max_chars:
                    remaining = max_chars - total
                    if remaining > 100:
                        passages.append(text[:remaining])
                    break
                passages.append(text)
                total += len(text)
                sources.append(
                    SourceReference(
                        book_name=meta["book_name"],
                        page_number=meta["page_number"],
                        source=meta["source"],
                        score=round(score, 4),
                        text_snippet=text[:200],
                    )
                )

            prompt = self._build_t5_prompt(question, passages)
            logger.debug(
                "Flan-T5 answering with {} passage(s) ({} chars).",
                len(passages),
                total,
            )
            answer_text = self._generate_with_t5(prompt)
            model_used = "flan-t5"

        logger.info(
            "Answer ready — model={}, chars={}, sources={}.",
            model_used,
            len(answer_text),
            len(sources),
        )

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            query=question,
            context_used=context_used,
            model_used=model_used,
        )

    def preload_models(self) -> None:
        """Eagerly load both models into memory at startup.

        Call this if you want to avoid first-query latency for both
        models.  Otherwise they load automatically on first use.
        """
        self._load_t5()
        self._load_bart()
