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

## Setup: Backend

### 1. Navigate to the backend directory

```bash
cd talal-rag-assistant/backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> First install may take several minutes — torch and transformers are large packages.

### 5. Configure local overrides if needed

```bash
# From the project root
cp .env.example .env
```

Defaults work without an environment file or API key. Copy the example only to customize backend settings.

---

## Setup: Frontend

### 1. Navigate to the frontend directory

```bash
cd talal-rag-assistant/frontend
```

### 2. Install Node dependencies

```bash
npm ci
```

---

## Running the Project

### Start the backend

```bash
# From backend/ with venv activated
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`

### Start the frontend

```bash
# From frontend/ in a separate terminal
npm start
```

Open http://localhost:3000. Vite forwards API requests to http://127.0.0.1:8000.
Set `API_PROXY_TARGET` before starting Vite if your backend runs on another address.
The first backend startup downloads the embedding model and indexes the bundled PDFs.
The first question and first summary each download their respective local model.

For a production bundle, run `npm run build` in `frontend/`. Serve `frontend/dist/`
and forward `/health` and `/api/` to FastAPI. If the frontend and API use
different origins, set `VITE_API_BASE_URL` at build time and add the frontend
origin to the backend's `ALLOWED_ORIGINS`. Browser-facing Vite values are public.

---

## Adding PDF Books

Select **Upload PDF** in the book library to add and index a text-based PDF.
You can also place PDFs in `backend/data/Books/` and restart the backend. The app supports both `Books` and `books`.
Use the library to browse the indexed titles and manage uploaded books.

---

## Running Tests

```bash
# From backend/ with venv activated
pytest tests/ -v
```

Run only unit tests:
```bash
pytest tests/unit/ -v
```

Run only integration tests:
```bash
pytest tests/integration/ -v
```

---

## Environment Variables

Backend overrides go in a local `.env` file at the repository root. Use `.env.example` as a template.

| Variable | Default | Description |
|---|---|---|
| `APP_PORT` | `8000` | Backend server port |
| `BOOKS_DIR` | `./data/books` | Directory containing PDF books |
| `CHROMA_DB_DIR` | `./data/chroma_db` | ChromaDB persistence directory |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_MODEL_NAME` | `google/flan-t5-large` | HuggingFace LLM |
| `LLM_MAX_NEW_TOKENS` | `512` | Max tokens in LLM response |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `RETRIEVER_TOP_K` | `4` | Number of chunks to retrieve per query |
| `TOPIC_SIMILARITY_THRESHOLD` | `0.30` | Cosine similarity cutoff for topic filter |

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
