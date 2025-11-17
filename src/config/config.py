# config/config.py

import os
import dotenv

dotenv.load_dotenv()

# --- OpenSearch Configuration ---
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "rag_documents_vectorstore_business_vv")

# --- Model Configuration ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# --- Data Ingestion Configuration ---
DATA_FILE_NAMES = ['ibm.json', 'myntra.json']
DATA_DIR = "data" # Your specified directory

# --- Constants for OpenSearch k-NN Mapping ---
EMBEDDING_DIMENSION = 768

# --- Vector Field Name ---
VECTOR_FIELD_NAME = "vector_field"

# --- Retrieval Configuration ---
K_VALUE = 5

# Ensure API Key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")