# /services/rag_api.py

# --- 1. Path Setup (MUST be first to resolve local imports) ---
import sys
import os
# This line must execute before 'config' is imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. Standard Library Imports (Alphabetical) ---
from pathlib import Path
import shutil
from typing import Annotated, Optional, List, Dict, Any

# --- 3. Third-party Library Imports (Alphabetical) ---
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

# --- 4. Local/Project-Specific Imports ---
# NOTE: DATA_FILE_NAMES is assumed to be imported from config.config
from config.config import DATA_DIR, DATA_FILE_NAMES, EMBEDDING_MODEL, INDEX_NAME 
from ingestion.ingest import get_opensearch_client, ingest_data_to_opensearch
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
    context: List[Context]

app = FastAPI(
    title="Risk RAG Retrieval API",
    description="Provides context-aware risk findings via Hybrid Search using OpenSearch and Google Embeddings."
)

# --- CORS MIDDLEWARE CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (WARNING: Restrict this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# --- NEW INGESTION ENDPOINT (Full Reindex from Configured Local Files) ---
@app.post("/ingest/configured_full", status_code=200)
def ingest_configured_data(
    # By default, force_reindex is True for a full re-initialization from local files
    force_reindex: Annotated[bool, Form()] = True 
):
    """
    Triggers a full reindex using the pre-configured list of files 
    (DATA_FILE_NAMES) from the local data path (DATA_DIR).
    This strictly uses local, pre-defined data paths.
    """
    try:
        print(f"\n--- API Triggered Full Reindex from Local Path (Files: {DATA_FILE_NAMES}) (Reindex: {force_reindex}) ---")
        
        # 1. Check if configured files exist
        missing_files = [f for f in DATA_FILE_NAMES if not Path(DATA_DIR).joinpath(f).exists()]
        if missing_files:
            raise HTTPException(
                status_code=404, 
                detail=f"Required configuration data files not found in {DATA_DIR}: {', '.join(missing_files)}"
            )

        # 2. Call the core ingestion function with the configured file list and directory
        ingest_data_to_opensearch(DATA_FILE_NAMES, DATA_DIR, force_reindex=force_reindex)

        return {
            "status": "Configured Ingestion Complete",
            "message": f"Successfully processed and indexed {len(DATA_FILE_NAMES)} files from local path {DATA_DIR} into index: {INDEX_NAME}.",
            "reindex_mode": force_reindex
        }
    except HTTPException:
        # Re-raise explicit HTTP exceptions (like 404)
        raise
    except Exception as e:
        print(f"Configured Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configured ingestion failed: {e}")


# --- ORIGINAL INGESTION ENDPOINT (Handles Uploaded File) ---
@app.post("/ingest/upload", status_code=200) 
async def ingest_full_data_from_upload(
    file: Annotated[UploadFile, File()], # The uploaded JSON file
    force_reindex: Annotated[bool, Form()] = False # The flag for reindexing
):
    """
    Uploads a new data file, saves it locally, and then runs the ingestion pipeline 
    on only that file. The file is saved to DATA_DIR.
    """
    if not file.filename or not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")

    # 1. Define the temporary save path within the DATA_DIR (e.g., data/uploaded_file.json)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    temp_file_path = Path(DATA_DIR) / file.filename 

    try:
        # Save the uploaded file content efficiently using shutil.copyfileobj
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Call the core ingestion function (only processes the single uploaded file)
        print(f"\n--- API Triggered Upload Ingestion: {file.filename} (Reindex: {force_reindex}) ---")
        
        # ingest_data_to_opensearch expects a list of filenames
        ingest_data_to_opensearch([file.filename], DATA_DIR, force_reindex=force_reindex)

        return {
            "status": "Ingestion Triggered",
            "message": f"Successfully processed and indexed data from uploaded file {file.filename} into index: {INDEX_NAME}. File remains saved in {DATA_DIR}.",
            "reindex_mode": force_reindex
        }
    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # FastAPI's UploadFile requires the underlying file to be closed, 
        # which happens implicitly when the request completes or explicitly via file.file.close()
        pass