# config.py

import os
import dotenv

dotenv.load_dotenv()

# --- OpenSearch Configuration ---
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "rag_documents_vectorstore")

# --- Model Configuration ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# --- Data Configuration ---
DATA_FILE = os.getenv("DATA_FILE", "data/knowledge-base.json")

# Ensure API Key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")