import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# --- OpenSearch Configuration ---
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "rag_documents_vectorstore_business_vv")

# --- Model Configuration ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-09-2025")

# --- Data Ingestion Configuration ---
DATA_FILE_NAMES = ['ibm.json', 'myntra.json']

# Robustly resolve the data directory relative to this config file
# If config.py is in src/config/, this points to src/../data (i.e., root/data)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

# --- Constants for OpenSearch k-NN Mapping ---
EMBEDDING_DIMENSION = 768
VECTOR_FIELD_NAME = os.getenv("VECTOR_FIELD_NAME", "vector_field")

# --- Retrieval Configuration ---
K_VALUE = int(os.getenv("K_VALUE", 5))

# Ensure API Key is set
if not os.getenv("GOOGLE_API_KEY"):
    # Make this a warning rather than a hard error to allow module imports in some environments
    print("WARNING: GOOGLE_API_KEY environment variable not set. Operations requiring LLMs will fail.")