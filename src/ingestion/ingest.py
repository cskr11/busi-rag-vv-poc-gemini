import os
import sys
import json
from collections import defaultdict

# Ensure the parent directory (src) is in the path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
# Document is now correctly aliased from langchain_core.documents
from langchain_core.documents import Document
from config.config import OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL, DATA_FILE

# --- CONSTANTS FOR OPENSEARCH k-NN MAPPING ---
EMBEDDING_DIMENSION = 768
VECTOR_FIELD_NAME = "vector_field"

# --- Helper Functions (Including updated mapping logic) ---

def get_opensearch_client():
    """Initializes and returns the raw OpenSearch client."""
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        use_ssl=False,
        verify_certs=False,
        request_timeout=30
    )

def delete_existing_index(client, index_name):
    """Deletes the OpenSearch index if it exists."""
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            print(f"-> Successfully deleted existing index: {index_name}")
        else:
            print(f"-> Index {index_name} does not exist. Skipping deletion.")
    except opensearch_exceptions.NotFoundError:
        print(f"-> Index {index_name} not found during deletion (expected).")
    except Exception as e:
        print(f"Error during index deletion: {e}")

def index_exists(client, index_name):
    """Checks if the OpenSearch index exists."""
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        print(f"Error checking index existence: {e}")
        return False

def create_index_with_knn_mapping(client, index_name, dimension, vector_field):
    """
    Explicitly maps metadata fields using dot notation (e.g., metadata.priority) 
    to ensure predictable searching/filtering.
    """
    print(f"  - Creating index {index_name} with k-NN mapping (dimension: {dimension})...")
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "l2",
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
                # Explicitly map the 'metadata' properties
                "metadata": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "keyword"},
                        "risk_category": {"type": "keyword"},
                        "risk_subcategory": {"type": "keyword"},
                        "priority": {"type": "integer"},
                        "doc_type": {"type": "keyword"},
                        "total_findings": {"type": "integer"}, 
                    }
                },
                # For keyword/text searchability
                "page_content": {"type": "text"}
            }
        }
    }
    try:
        client.indices.create(index=index_name, body=index_body)
        print(f"  - Successfully created index: {index_name}")
    except opensearch_exceptions.RequestError as e:
        if 'resource_already_exists_exception' in str(e):
            print(f"  - Index {index_name} already exists. Skipping creation.")
        else:
            print(f"Error creating index: {e}")
            raise

# ----------------------------------------------------------------------
# --- UPDATED DATA PROCESSING FUNCTION ---
# ----------------------------------------------------------------------

def process_risk_data_to_documents(data):
    """
    Correctly parses risk data, extracts company name, 
    and generates separate Document objects for each risk finding and the overall summary.
    """
    print("  - Processing complex JSON into Document objects...")
    document_list = []
    
    for risk_profile in data.get('payload', {}).get('documents', []):
        doc_id = risk_profile.get('documentId')
        identity_doc = risk_profile.get('sourceDocument', {}).get('identitySourceDocument', {})
        
        # 1. Extract Primary Company Name 
        business_names_identifiers = identity_doc.get('businessNameIdentifiers', [])
        business_names = [
            identifier['identifier']['data']['value']
            for identifier in business_names_identifiers
        ]
        # Use the first business name as the primary name
        primary_business_name = business_names[0] if business_names else 'N/A'
        
        # Extract primary address line for context
        address_info = identity_doc.get('addressIdentifiers', [{}])[0].get('identifier', {}).get('data', {})
        business_address = address_info.get('line1', 'Unknown Address')

        # 2. Process each specific Risk Finding
        risk_findings = risk_profile.get('riskFindings', [])
        risk_categories_present = set()

        for finding in risk_findings:
            risk_category = finding.get('riskCategory', 'N/A')
            # The JSON for H1B subcategory has a trailing space, trimming it here for cleaner filtering later
            risk_subcategory = finding.get('riskSubcategory', 'N/A').strip() 
            risk_priority = finding.get('priority', 99) # Default to 99 if not found
            
            risk_categories_present.add(risk_category)

            # Document content for retrieval: focuses on the finding text and context
            content_text = (
                f"Company: {primary_business_name} (ID: {doc_id})\n"
                f"Risk Finding: {finding.get('findingText', 'N/A')}\n"
                f"Category: {risk_category} (Subcategory: {risk_subcategory})\n"
                f"Priority: {risk_priority}"
            )
            
            # Metadata for filtering and precise context
            metadata = {
                "business_id": doc_id, 
                "company_name": primary_business_name,
                "doc_type": "RiskFinding",
                "risk_category": risk_category, 
                "risk_subcategory": risk_subcategory,
                "priority": risk_priority, 
                "source": finding.get('source'),
                "explanation_snippet": str(finding.get('explanations', []))
            }
            document_list.append(Document(page_content=content_text, metadata=metadata))

        # 3. Generate the Profile Summary Document (for high-level queries)
        profile_score = risk_profile.get('profileScoreDetail', {}).get('profileScore', 0.0)
        
        summary_content = (
            f"Overall risk profile summary for **{primary_business_name}** located at {business_address}.\n"
            f"Business ID: {doc_id}\n"
            f"Overall Profile Score: {profile_score}\n"
            f"Total Detailed Risk Findings: {len(risk_findings)}\n"
            f"Risk Categories Present: {', '.join(sorted(list(risk_categories_present))) if risk_categories_present else 'None'}"
        )
        summary_metadata = {
            "business_id": doc_id, 
            "company_name": primary_business_name,
            "doc_type": "ProfileSummary", 
            "profile_score": profile_score, 
            "total_findings": len(risk_findings),
            "risk_categories_summary": ', '.join(sorted(list(risk_categories_present)))
        }
        document_list.append(Document(page_content=summary_content, metadata=summary_metadata))

    print(f"  - Generated {len(document_list)} Document objects.")
    return document_list

# --- Core Ingestion Logic ---

def _load_and_process_file(file_path):
    """Loads JSON file and transforms data into Document objects, handling errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return process_risk_data_to_documents(raw_data)
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: File not found or is not valid JSON at {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue with file {file_path}. {e}")
        print("  - Please ensure the file is saved as UTF-8.")
        return None
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return None

def _run_opensearch_ingestion(documents_to_ingest, os_client, embeddings):
    """Internal function to execute the OpenSearch indexing command."""
    try:
        # Use the Document from langchain_core.documents
        from langchain_core.documents import Document as LCDocument 
        
        # Ensure we are passing standard LangChain Documents
        lcd_documents = [
            LCDocument(page_content=doc.page_content, metadata=doc.metadata) 
            for doc in documents_to_ingest
        ]

        OpenSearchVectorSearch.from_documents(
            lcd_documents, embeddings, opensearch_url=OPENSEARCH_URL,
            index_name=INDEX_NAME, client=os_client,
            vector_field=VECTOR_FIELD_NAME, 
            # Disable kwargs that might cause issues if client handling changes
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        print(f"-> Successfully indexed {len(documents_to_ingest)} documents into OpenSearch index: {INDEX_NAME}")
        return True
    except Exception as e:
        print(f"Error connecting to or indexing in OpenSearch at {OPENSEARCH_URL}. Is OpenSearch running? Error: {e}")
        return False

# --- Main Logic Function for Ingestion ---

def ingest_data_to_opensearch(file_path, force_reindex=False):
    """Handles initial indexing or full reindexing."""
    os_client = get_opensearch_client()

    if index_exists(os_client, INDEX_NAME):
        if force_reindex:
            print("--- FORCE REINDEX: Deleting existing index and reloading. ---")
            delete_existing_index(os_client, INDEX_NAME)
        else:
            print(f"--- Index {INDEX_NAME} already exists. Skipping full ingestion. ---")
            return

    # Check/Create index AFTER deletion (if applicable)
    if not index_exists(os_client, INDEX_NAME):
        try:
            create_index_with_knn_mapping(os_client, INDEX_NAME, EMBEDDING_DIMENSION, VECTOR_FIELD_NAME)
        except Exception as e:
            print(f"Failed to create index mapping: {e}")
            return

    print(f"\n-> Starting data ingestion process into index: {INDEX_NAME}")
    documents_to_ingest = _load_and_process_file(file_path)
    if not documents_to_ingest: return
    
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"  - Initialized Google Embeddings model: {EMBEDDING_MODEL}")
    
    _run_opensearch_ingestion(documents_to_ingest, os_client, embeddings)

def append_data_to_opensearch(file_path):
    """Appends new data to an existing OpenSearch index."""
    os_client = get_opensearch_client()
    
    if not index_exists(os_client, INDEX_NAME):
        print(f"--- Index {INDEX_NAME} not found. Running initial index creation before appending. ---")
        try:
            create_index_with_knn_mapping(os_client, INDEX_NAME, EMBEDDING_DIMENSION, VECTOR_FIELD_NAME)
        except Exception as e:
            print(f"Failed to create index mapping, cannot append: {e}")
            return
    
    print(f"\n-> Starting incremental data APPEND process into index: {INDEX_NAME}")
    documents_to_ingest = _load_and_process_file(file_path)
    if not documents_to_ingest: return
    
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"  - Initialized Google Embeddings model: {EMBEDDING_MODEL}")
    
    _run_opensearch_ingestion(documents_to_ingest, os_client, embeddings)

# You will need to maintain your __main__ block separately, calling one of the
# ingestion functions based on your desired mode (e.g., ingest_data_to_opensearch(DATA_FILE, force_reindex=True)).

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Example: Run a full reindex to ensure a clean start
    print("\n" + "="*50)
    print("MODE: FULL REINDEX (Deletes and recreates index)")
    ingest_data_to_opensearch(DATA_FILE, force_reindex=True)