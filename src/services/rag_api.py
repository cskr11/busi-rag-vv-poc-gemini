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
from fastapi.middleware.cors import CORSMiddleware # IMPORT ADDED HERE
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

# --- 4. Local/Project-Specific Imports ---
# These imports now work because the path was appended in step 1.
from config.config import DATA_DIR, EMBEDDING_MODEL, INDEX_NAME 
# NOTE: Removed append_data_to_opensearch import as it is missing from ingest.py
from ingestion.ingest import get_opensearch_client, ingest_data_to_opensearch
# Alias the functioning simple_hybrid_retriever to the expected name
from retrieval.query import simple_hybrid_retriever as hybrid_retriever 


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

# --- CORS MIDDLEWARE CONFIGURATION ADDED HERE ---
# Allowing all origins, headers, and methods for easy local/POC testing.
# WARNING: Restrict 'allow_origins' in production for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --------------------------------------------------

try:
    # Initialize shared clients globally
    OS_CLIENT = get_opensearch_client()
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print("API Clients initialized successfully.")
except Exception as e:
    # This prevents the API from starting if core database/embedding services are unavailable
    print(f"FATAL: Could not initialize clients. Is OpenSearch running? Error: {e}")
    raise Exception("Service failed to initialize core clients.")


# --- API Endpoint (Retrieval) ---
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
    retrieved_docs = hybrid_retriever(user_query, OS_CLIENT, EMBEDDINGS)

    # 2. Format LangChain Documents for API response
    context_list = [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in retrieved_docs
    ]

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

# --- INGESTION ENDPOINT (Full Reindex) ---
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
    if not file.filename or not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")

    # 1. Define the temporary save path within the DATA_DIR (e.g., data/uploaded.json)
    temp_file_path = Path(DATA_DIR) / file.filename 

    try:
        # Save the uploaded file content
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Call the core ingestion function (expects a list of filenames and directory)
        print(f"\n--- API Triggered Ingestion: {file.filename} (Reindex: {force_reindex}) ---")
        
        # ingest_data_to_opensearch expects (file_names_list, data_dir, force_reindex)
        ingest_data_to_opensearch([file.filename], DATA_DIR, force_reindex=force_reindex)

        return {
            "status": "Ingestion Triggered",
            "message": f"Successfully processed and indexed data from {file.filename} into index: {INDEX_NAME}.",
            "reindex_mode": force_reindex
        }
    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # 3. Clean up the temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()


# # --- NEW APPEND ENDPOINT (Incremental Updates) ---
# @app.post("/ingest/append")
# async def ingest_append_data(
#     file: Annotated[UploadFile, File()]
# ):
#     """
#     [COMMENTED OUT] This endpoint requires the 'append_data_to_opensearch' function 
#     to be implemented in ingestion/ingest.py.
#     """
#     # NOTE: If implementing later, use the same logic as /ingest/full but call 
#     # append_data_to_opensearch([file.filename], DATA_DIR)
#     raise HTTPException(status_code=501, detail="Incremental ingestion endpoint not implemented in backend.")