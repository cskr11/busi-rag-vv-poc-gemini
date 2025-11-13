# config.py

import os

# --- OpenSearch Configuration ---
OPENSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "rag_documents"

# --- Model Configuration ---
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

# --- Data Configuration ---
DATA_FILE = "data.json"

# Ensure API Key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")