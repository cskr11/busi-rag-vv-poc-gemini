# /services/rag_api.py
import sys
import os
import asyncio
from pathlib import Path
import shutil

# --- Setup path to resolve 'config', 'ingestion', 'retrieval' ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Local Imports
# FIX: Import K_VALUE for Issue 4 fix
from config.config import DATA_DIR, DATA_FILE_NAMES, EMBEDDING_MODEL, INDEX_NAME, K_VALUE
from ingestion.ingest import get_opensearch_client, ingest_data_to_opensearch
from retrieval.query import simple_hybrid_retriever, condense_query, get_llm_response

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str # 'user' or 'model'
    text: str

class RetrievalRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = [] # Type hint allows None, default is empty list
    filters: Optional[Dict[str, Any]] = None
    k: Optional[int] = 5

class Context(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    query: str
    search_query: str
    response: str
    context: List[Context]

# --- App & Global Clients ---
app = FastAPI(title="Context-Aware Risk RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Globals
try:
    OS_CLIENT = get_opensearch_client()
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("✅ Core clients initialized.")
except Exception as e:
    print(f"❌ Failed to initialize clients: {e}")
    OS_CLIENT = None
    EMBEDDINGS = None


@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: RetrievalRequest):
    """
    Performs context-aware RAG, using history to refine the search query,
    and returns a final LLM-generated response.
    """
    if not OS_CLIENT or not EMBEDDINGS:
        raise HTTPException(status_code=503, detail="Search services unavailable.")

    # FIX: Ensure request.history is treated as an empty list if it is None.
    safe_history = request.history if request.history is not None else []
    history_list = [{"role": msg.role, "text": msg.text} for msg in safe_history]

    # FIX (Issue 4): Coalesce request.k to K_VALUE if it is None
    retrieval_k = request.k if request.k is not None else K_VALUE

    # 1. Condense Query: Make the search context-aware
    search_query = await condense_query(history_list, request.query)

    # 2. Retrieve Documents using the condensed query
    docs = simple_hybrid_retriever(
        search_query=search_query,
        client=OS_CLIENT,
        embeddings=EMBEDDINGS,
        k=retrieval_k, # Use the guaranteed int k value
        filters=request.filters
    )

    if not docs:
        return ChatResponse(
            query=request.query,
            search_query=search_query,
            response="I couldn't find any relevant risk documents for your request.",
            context=[]
        )

    # 3. Format Context for LLM
    # FIX (Issue 3): Explicitly construct Pydantic Context objects from the dict list
    context_list = [
        Context(content=d.page_content, metadata=d.metadata)
        for d in docs
    ]
    context_str = "\n".join([
        f"[Doc {i+1} from {d.metadata.get('file_source_tag', 'N/A')}] {d.page_content.strip().replace('\n', ' ')}"
        for i, d in enumerate(docs)
    ])

    # 4. Generate Final LLM Response (Pass the original query for final instruction)
    response_text = await get_llm_response(request.query, context_str)

    return ChatResponse(
        query=request.query,
        search_query=search_query,
        response=response_text,
        context=context_list # context_list is now List[Context]
    )


# --- Ingestion Endpoints (Fixed for upload safety) ---
@app.post("/ingest/configured_full")
def ingest_configured_data(force_reindex: bool = Form(True)):
    """Re-indexes the default dataset (ibm.json, myntra.json)."""
    try:
        # Verify files exist
        missing = [f for f in DATA_FILE_NAMES if not os.path.exists(os.path.join(DATA_DIR, f))]
        if missing:
            raise HTTPException(404, detail=f"Missing files in {DATA_DIR}: {missing}")

        ingest_data_to_opensearch(DATA_FILE_NAMES, DATA_DIR, force_reindex=force_reindex)
        return {"status": "Success", "message": f"Indexed {len(DATA_FILE_NAMES)} files."}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...), force_reindex: bool = Form(False)):
    """Uploads and indexes a new JSON file."""

    filename = file.filename

    # FIX (Issue 1 & 2): Safely check filename and ensure it's not None
    if not filename:
        raise HTTPException(400, detail="Filename missing for uploaded file.")

    if not filename.lower().endswith('.json'):
        raise HTTPException(400, detail="Only JSON files allowed.")

    # FIX (Issue 2): filename is now guaranteed to be a string
    file_path = os.path.join(DATA_DIR, filename)

    try:
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ingest_data_to_opensearch([filename], DATA_DIR, force_reindex=force_reindex)

        return {"status": "Success", "file": filename, "message": "Indexed successfully."}
    except Exception as e:
        raise HTTPException(500, detail=f"Ingestion failed: {e}")