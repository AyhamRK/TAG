"""
core/config.py
==============
Centralised application settings loaded from the ``.env`` file using
``pydantic-settings``.

All environment variables are documented with types and defaults.
Import the ``settings`` singleton anywhere in the app:

    from app.core.config import settings
    print(settings.app_title)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration.

    Values are read (in priority order) from:
    1. Real environment variables
    2. The ``.env`` file at the project root
    3. The defaults defined here

    Attributes
    ----------
    app_title : str
        Human-readable API title shown in the Swagger UI.
    app_version : str
        Semantic version string.
    app_env : str
        ``"development"`` or ``"production"``.
    app_debug : bool
        Enable FastAPI debug mode.
    app_host : str
        Bind host for uvicorn.
    app_port : int
        Bind port for uvicorn.
    allowed_origins : str
        Comma-separated list of CORS origins (e.g. ``"http://localhost:3000"``).
    pdf_data_dir : Path
        Directory where uploaded PDFs are saved permanently.
    books_dir : Path
        Directory scanned for books on startup (same as pdf_data_dir by default).
    chroma_db_dir : Path
        Directory for the ChromaDB persistent store.
    embedding_model_name : str
        HuggingFace sentence-transformer model identifier.
    embedding_device : str
        PyTorch device for embeddings (``"cpu"`` or ``"cuda"``).
    llm_model_name : str
        HuggingFace model identifier for the generative LLM.
    llm_device : str
        PyTorch device for the LLM.
    llm_max_new_tokens : int
        Maximum tokens the LLM may generate per response.
    llm_temperature : float
        Sampling temperature (0 = deterministic greedy).
    chunk_size : int
        Character length of each text chunk.
    chunk_overlap : int
        Overlap in characters between adjacent chunks.
    retriever_top_k : int
        Number of chunks retrieved per query.
    chroma_collection_name : str
        Name of the ChromaDB collection.
    topic_similarity_threshold : float
        Minimum cosine similarity for the guardrail to allow a question.
    log_level : str
        Loguru log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────
    app_title: str = "Talal Abu Ghazaleh AI Book Assistant"
    app_version: str = "1.0.0"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ── CORS ──────────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:3000"

    # ── Paths ─────────────────────────────────────────────────────────
    pdf_data_dir: Path = Path(__file__).resolve().parents[2] / "data" / "books"
    books_dir: Path = Path(__file__).resolve().parents[2] / "data" / "books"
    chroma_db_dir: Path = Path(__file__).resolve().parents[2] / "data" / "chroma_db"

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ── LLM ──────────────────────────────────────────────────────────
    llm_model_name: str = "google/flan-t5-large"
    summariser_model_name: str = "facebook/bart-large-cnn"
    llm_device: str = "cpu"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.1

    # ── RAG ──────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_top_k: int = 4
    chroma_collection_name: str = "talal_books"
    topic_similarity_threshold: float = 0.30

    # ── Logging ───────────────────────────────────────────────────────
    log_level: str = "INFO"

    @property
    def cors_origins(self) -> list[str]:
        """Return CORS origins as a list (split on comma).

        Returns
        -------
        list[str]
            Parsed list of allowed origin URLs.
        """
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance.

    Uses ``lru_cache`` so the ``.env`` file is read exactly once per
    process, regardless of how many times this function is called.

    Returns
    -------
    Settings
        The application settings object.
    """
    return Settings()


# Convenient module-level singleton for direct imports.
settings: Settings = get_settings()
