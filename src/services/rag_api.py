# /services/rag_api.py

# --- 1. Path Setup (MUST be first to resolve local imports) ---
import sys
import os
# This line must execute before 'config' is imported.
# It adds the project root (where 'config' is) to the Python search path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. Standard Library Imports (Alphabetical) ---
from pathlib import Path
import shutil
from typing import Annotated, Optional, List, Dict, Any

# --- 3. Third-party Library Imports (Alphabetical) ---
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

# --- 4. Local/Project-Specific Imports ---
# These imports now work because the path was appended in step 1.
from config.config import DATA_FILE, EMBEDDING_MODEL
from ingestion.ingest import append_data_to_opensearch, get_opensearch_client, ingest_data_to_opensearch
from retrieval.query import hybrid_retriever
# Define the expected output structure using Pydantic's BaseModel
class Context(BaseModel):
    # Defines the structure of a single retrieved chunk
    content: str
    metadata: Dict[str, Any]

class ContextResponse(BaseModel):
    # Defines the overall response structure
    query: str
    status: str
    count: int
    context: List[Context] # <-- Use the Pydantic model here

app = FastAPI(
    title="Risk RAG Retrieval API",
    description="Provides context-aware risk findings via Hybrid Search using OpenSearch and Google Embeddings."
)

try:
    # Initialize shared clients globally
    OS_CLIENT = get_opensearch_client()
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("API Clients initialized successfully.")
except Exception as e:
    # This prevents the API from starting if core database/embedding services are unavailable
    print(f"FATAL: Could not initialize clients. Is OpenSearch running? Error: {e}")
    raise Exception("Service failed to initialize core clients.")


# --- API Endpoint ---
@app.post("/retrieve", response_model=ContextResponse)
def retrieve_context(body: dict):
    """
    Accepts a user query and returns the top relevant documents (chunks)
    using the Hybrid Search logic (Vector Similarity + Metadata Filters).
    """
    user_query = body.get("query")

    if not user_query:
        raise HTTPException(status_code=400, detail="The 'query' field is required in the request body.")

    # 1. Execute the Hybrid Search using the imported function
    # The hybrid_retriever returns a list of LangChain Document objects
    retrieved_docs = hybrid_retriever(user_query, OS_CLIENT, EMBEDDINGS)

    # 2. Format LangChain Documents for API response
    context_list = []
    for doc in retrieved_docs:
        context_list.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    if not context_list:
        return {
            "query": user_query,
            "status": "No results found.",
            "count": 0,
            "context": []
        }

    return {
        "query": user_query,
        "status": "Success",
        "count": len(context_list),
        "context": context_list
    }

# --- NEW INGESTION ENDPOINT ---
@app.post("/ingest/full")
async def ingest_full_data(
    file: Annotated[UploadFile, File()], # The uploaded JSON file
    force_reindex: Annotated[bool, Form()] = False # The flag for reindexing
):
    """
    Uploads a new data file and runs the full ingestion pipeline.

    Accepts:
    - file: The JSON file containing the risk data.
    - force_reindex: If True, deletes and recreates the index before loading.
    """
    filename_parts = file.filename.split('.') if file.filename else []
    if not filename_parts or filename_parts[-1].lower() != 'json':
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")

    # 1. Save the uploaded file temporarily
    # We use a temporary file path from the source data dir for ingestion logic compatibility
    temp_file_path = Path(DATA_FILE)

    try:
        # FastAPI's UploadFile uses a file-like object; copy its content to a temp file
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Call the core ingestion function
        print(f"\n--- API Triggered Ingestion: {file.filename} (Reindex: {force_reindex}) ---")
        ingest_data_to_opensearch(str(temp_file_path), force_reindex=force_reindex)

        return {
            "status": "Ingestion Triggered",
            "message": f"Successfully processed and indexed data from {file.filename}.",
            "reindex_mode": force_reindex
        }
    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # 3. Clean up the temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()

# --- NEW APPEND ENDPOINT (Optional, for incremental updates) ---
@app.post("/ingest/append")
async def ingest_append_data(
    file: Annotated[UploadFile, File()]
):
    """
    Uploads a new data file and appends documents to the existing index.
    """
    filename_parts = file.filename.split('.') if file.filename else []
    if not filename_parts or filename_parts[-1].lower() != 'json':
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")

    temp_file_path = Path(DATA_FILE)

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"\n--- API Triggered Append: {file.filename} ---")
        append_data_to_opensearch(str(temp_file_path))

        return {
            "status": "Append Triggered",
            "message": f"Successfully appended data from {file.filename} to index.",
        }
    except Exception as e:
        print(f"Append failed: {e}")
        raise HTTPException(status_code=500, detail=f"Append failed: {e}")
    finally:
        pass
        # if temp_file_path.exists():
        #     temp_file_path.unlink()