# Talal Abu Ghazaleh AI Book Assistant

An intelligent question-answering assistant that reads books authored by **Talal Abu Ghazaleh** and answers questions **strictly based on their content**. Built as a final internship project using a fully local RAG (Retrieval-Augmented Generation) pipeline — no external API calls.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Setup: Backend](#setup-backend)
- [Setup: Frontend](#setup-frontend)
- [Running the Project](#running-the-project)
- [Adding PDF Books](#adding-pdf-books)
- [Running Tests](#running-tests)
- [Environment Variables](#environment-variables)
- [Tech Stack](#tech-stack)

---

## Project Overview

This assistant uses a **RAG pipeline** to:

1. Parse PDF books by Talal Abu Ghazaleh using **PyMuPDF**
2. Split text into chunks and embed them using **sentence-transformers/all-MiniLM-L6-v2**
3. Store and retrieve chunks from a local **ChromaDB** vector store
4. Answer questions using a local **Flan-T5-Large** model via **LangChain**
5. Reject questions unrelated to the book content entirely

The frontend is a clean **React** chat interface that communicates with a **FastAPI** backend.

---

## Architecture

```
User Question
     │
     ▼
[Topic Filter] ──── Off-topic ──▶ Rejection Response
     │
  On-topic
     │
     ▼
[Embeddings: all-MiniLM-L6-v2]
     │
     ▼
[ChromaDB Vector Store] ──▶ Top-K Relevant Chunks
     │
     ▼
[LangChain QA Chain + Flan-T5-Large]
     │
     ▼
Answer (grounded in book content only)
```

---

## Folder Structure

```
talal-rag-assistant/
│
├── backend/                        # Python + FastAPI application
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/             # API route handlers (chat, ingest, health)
│   │   ├── core/                   # Config, settings, startup logic
│   │   ├── db/                     # ChromaDB client and collection management
│   │   ├── models/                 # Pydantic request/response schemas
│   │   ├── services/               # Business logic (RAG, embeddings, LLM, topic filter)
│   │   └── utils/                  # Shared helpers (PDF parsing, text splitting, logging)
│   ├── data/
│   │   ├── Books/                  # Source PDFs indexed on startup
│   │   └── chroma_db/              # Persistent ChromaDB store (git-ignored)
│   ├── tests/
│   │   ├── unit/                   # Unit tests for individual services
│   │   └── integration/            # Integration tests for API endpoints
│   ├── requirements.txt
│   └── main.py                     # FastAPI app entry point
│
├── frontend/                       # React chat interface
│   ├── index.html                   # Vite application entry
│   └── src/                         # Chat, citations, and book library
│
├── docs/                           # Additional documentation and diagrams
├── .env.example                    # Template for environment variables
├── .gitignore
└── README.md
```

---

## Prerequisites

Make sure you have the following installed:

| Tool | Minimum Version | Check |
|---|---|---|
| Python | 3.11 recommended | `python --version` |
| Node.js | 20+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | any | `git --version` |

> **Note:** The LLM and embedding models run entirely on CPU by default. A machine with at least **8 GB RAM** is recommended. 16 GB is ideal for comfortable performance.

---


## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (local) |
| LLM | google/flan-t5-large (HuggingFace, local) |
| Orchestration | LangChain |
| Frontend | React 18 |
| HTTP client | Axios |
| Testing | pytest + httpx |

---

## Notes

- All AI models run **fully locally** — no internet connection is required after initial model download.
- The assistant will **refuse to answer** questions unrelated to the loaded book content.
- Models are cached by HuggingFace in `~/.cache/huggingface/` after first download.
