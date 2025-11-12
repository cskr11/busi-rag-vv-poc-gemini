import os
import json
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # <-- FIXED IMPORT
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/knowledge_base.json"
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "rag_knowledge_index"
# The recommended Gemini embedding model
EMBEDDING_MODEL = "models/text-embedding-004" 

def load_and_index_data():
    """Loads JSON, chunks it, creates embeddings, and stores in OpenSearch."""
    
    # --- 1. Load Documents ---
    # jq_schema selects the content to load. Here, it selects the 'content' field 
    # from each object in the root array.
    print("Loading documents...")
    loader = JSONLoader(
        file_path=DATA_PATH,
        jq_schema='.[].content', 
        text_content=True,
    )
    documents = loader.load()

    # --- 2. Split Documents ---
    print(f"Loaded {len(documents)} initial documents. Splitting into chunks...")
    # Use the corrected class name
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # --- 3. Create Embeddings & Store in OpenSearch ---
    print("Creating embeddings and connecting to OpenSearch...")
    
    # Initialize the Gemini Embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Initialize OpenSearch Client 
    # NOTE: Adjust http_auth if your OpenSearch requires authentication
    os_client = OpenSearch(
        hosts=[OPENSEARCH_URL],
        http_auth=('user', 'password') 
    )
    
    # Store documents/vectors in OpenSearch
    OpenSearchVectorSearch.from_documents(
        chunks,
        embeddings,
        opensearch_url=OPENSEARCH_URL,
        http_auth=('user', 'password'), # Adjust auth
        index_name=INDEX_NAME,
        client=os_client,
        engine="nmslib", # Example engine
    )
    print(f"Successfully indexed {len(chunks)} chunks into OpenSearch index: {INDEX_NAME}")

if __name__ == "__main__":
    load_and_index_data()