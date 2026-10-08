from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.config import Settings
from app.embeddings import OpenAIEmbedder
from app.generation import OpenAIGenerator, RagService
from app.ingestion import chunk_text
from app.models import AskRequest, AskResponse, DocumentInput, IngestResponse
from app.retrieval import FaissVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
INDEX_PATH = Path("data/index.faiss")
METADATA_PATH = Path("data/chunks.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required. Copy .env.example to .env and set it.")
    store = FaissVectorStore()
    store.load(INDEX_PATH, METADATA_PATH)
    app.state.settings = settings
    app.state.service = RagService(
        OpenAIEmbedder(settings.openai_api_key, settings.embedding_model),
        OpenAIGenerator(settings.openai_api_key, settings.chat_model),
        store,
        settings.top_k,
    )
    yield
    store.save(INDEX_PATH, METADATA_PATH)


app = FastAPI(title="Inspectable RAG Document Q&A Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, int | str]:
    service: RagService = request.app.state.service
    return {"status": "ok", "indexed_chunks": service.store.size}


@app.post("/documents", response_model=IngestResponse, status_code=201)
def ingest(document: DocumentInput, request: Request) -> IngestResponse:
    settings: Settings = request.app.state.settings
    service: RagService = request.app.state.service
    chunks = chunk_text(document.text, document.filename, settings.chunk_size, settings.chunk_overlap)
    try:
        added = service.index(chunks)
    except Exception as exc:
        logging.exception("Document indexing failed")
        raise HTTPException(status_code=502, detail="Embedding provider failed") from exc
    return IngestResponse(chunks_added=added, total_chunks=service.store.size)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    service: RagService = request.app.state.service
    try:
        return service.ask(payload.question, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Question answering failed")
        raise HTTPException(status_code=502, detail="LLM provider failed") from exc

