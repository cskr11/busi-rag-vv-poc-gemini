# ingest.py

import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions

# Document is used to structure the data for ingestion
from langchain_core.documents import Document

# Configuration
from config import DATA_FILE, OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL

# --- CONSTANTS FOR OPENSEARCH k-NN MAPPING ---
# NOTE: This must match the dimension of the EMBEDDING_MODEL (e.g., 768 for text-embedding-004)
EMBEDDING_DIMENSION = 768  # <-- **DIMENSION ADDED HERE**
VECTOR_FIELD_NAME = "vector_field" # Default field name used by LangChain OpenSearch connector

# --- Helper Functions (Unchanged) ---

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

# --- New Index Creation Function (ADDED) ---
def create_index_with_knn_mapping(client, index_name, dimension, vector_field):
    """Creates the OpenSearch index with the k-NN vector field mapping."""
    print(f"   - Creating index {index_name} with k-NN mapping (dimension: {dimension})...")
    index_body = {
        "settings": {
            "index": {
                "knn": True,  # Enable k-NN functionality
                "knn.space_type": "l2",
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,  # <-- DIMENSION SPECIFIED HERE
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
                "metadata": {"type": "object"} # Ensure metadata is indexed properly
            }
        }
    }
    try:
        client.indices.create(index=index_name, body=index_body)
        print(f"   - Successfully created index: {index_name}")
    except opensearch_exceptions.RequestError as e:
        if 'resource_already_exists_exception' in str(e):
             print(f"   - Index {index_name} already exists. Skipping creation.")
        else:
             print(f"Error creating index: {e}")
             raise

# --- JSON Processing Function (Unchanged) ---

def process_risk_data_to_documents(data):
    """
    Transforms the complex risk JSON into a list of Document objects
    (one for each risk finding and one for the summary).
    """
    print("   - Processing complex JSON into Document objects...")
    document_list = []

    # Assumes the structure from your file: payload -> documents (list)
    for risk_profile in data.get('payload', {}).get('documents', []):
        doc_id = risk_profile.get('documentId')
        source_doc = risk_profile.get('sourceDocument', {})
        identity_doc = source_doc.get('identitySourceDocument', {})

        # Get base identity info
        business_id = doc_id
        business_address = identity_doc.get('addressIdentifiers', [{}])[0].get('identifier', {}).get('data', {}).get('line1', 'Unknown Address')

        # STRUCTURE 1: Create a Document for each Risk Finding
        risk_findings = source_doc.get('riskFindings', [])
        for finding in risk_findings:
            # Create the text to be vectorized (the 'page_content')
            content_text = (
                f"Risk Finding: {finding.get('findingText', 'N/A')}\n"
                f"Category: {finding.get('riskCategory', 'N/A')} (Subcategory: {finding.get('riskSubcategory', 'N/A')})\n"
                f"Priority: {finding.get('priority', 'N/A')}\n"
                f"Context: {finding.get('findingContext', 'N/A')}\n"
                f"Source: {finding.get('source', 'N/A')}"
            )

            # Create the metadata for filtering
            metadata = {
                "business_id": business_id,
                "business_address": business_address,
                "doc_type": "RiskFinding",
                "risk_category": finding.get('riskCategory'),
                "risk_subcategory": finding.get('riskSubcategory'),
                "priority": finding.get('priority'),
                "source": finding.get('source')
            }

            document_list.append(Document(page_content=content_text, metadata=metadata))

        # STRUCTURE 2: Create one Document for the Profile Summary
        profile_score = source_doc.get('profileScoreDetail', {}).get('profileScore', 'N/A')

        summary_content = (
            f"Overall risk profile summary for business at {business_address}.\n"
            f"Business ID: {business_id}\n"
            f"Overall Profile Score: {profile_score}\n"
            f"Total Detailed Risk Findings: {len(risk_findings)}"
        )

        summary_metadata = {
            "business_id": business_id,
            "business_address": business_address,
            "doc_type": "ProfileSummary",
            "profile_score": profile_score,
            "total_findings": len(risk_findings)
        }
        document_list.append(Document(page_content=summary_content, metadata=summary_metadata))

    print(f"   - Generated {len(document_list)} Document objects.")
    return document_list

# --- Core Ingestion Logic (Modified) ---

def ingest_data_to_opensearch(file_path, force_reindex=False):
    """
    Handles data ingestion from the complex JSON file.
    """
    os_client = get_opensearch_client()

    if index_exists(os_client, INDEX_NAME):
        if force_reindex:
            print("--- FORCE REINDEX: Deleting existing index and reloading. ---")
            delete_existing_index(os_client, INDEX_NAME)
        else:
            print(f"--- Index {INDEX_NAME} already exists. Skipping ingestion. ---")
            return

    # Check existence again after potential deletion
    if not index_exists(os_client, INDEX_NAME):
        # 1. CREATE INDEX with k-NN MAPPING (New Step)
        try:
             create_index_with_knn_mapping(os_client, INDEX_NAME, EMBEDDING_DIMENSION, VECTOR_FIELD_NAME)
        except Exception as e:
            print(f"Failed to create index mapping: {e}")
            return


    print(f"\n-> Starting data ingestion process into index: {INDEX_NAME}")

    # 2. Load Data and Transform into Documents (Same as original step 1)
    try:
        #
        # --- THIS IS THE FIX ---
        # Specify UTF-8 encoding to handle special characters
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        # --- END OF FIX ---
        #

        # Use our new processing function
        documents_to_ingest = process_risk_data_to_documents(raw_data)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: File not found or is not valid JSON at {file_path}: {e}")
        return
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue with file {file_path}. {e}")
        print("   - Please ensure the file is saved as UTF-8.")
        return
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return

    # 3. Initialize Embeddings (Same as original step 2)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"   - Initialized Google Embeddings model: {EMBEDDING_MODEL}")

    # 4. Ingest Documents (Same as original step 3)
    try:
        OpenSearchVectorSearch.from_documents(
            documents_to_ingest,
            embeddings,
            opensearch_url=OPENSEARCH_URL,
            index_name=INDEX_NAME,
            # Pass the client so LangChain doesn't try to create the index again
            client=os_client,
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        print(f"-> Successfully indexed {len(documents_to_ingest)} documents into OpenSearch index: {INDEX_NAME}")
    except Exception as e:
        print(f"Error connecting to or indexing in OpenSearch at {OPENSEARCH_URL}. Is OpenSearch running? Error: {e}")


# --- MAIN EXECUTION (Unchanged) ---

if __name__ == "__main__":
    # Run the ingestion with force_reindex=True to ensure it loads fresh data
    ingest_data_to_opensearch(DATA_FILE, force_reindex=True)