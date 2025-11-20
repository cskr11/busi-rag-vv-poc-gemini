import os
import sys
import json
from typing import List, Dict, Any

# --- Setup Paths ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
from langchain_core.documents import Document
from config.config import (
    DATA_DIR, DATA_FILE_NAMES, OPENSEARCH_URL, INDEX_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, VECTOR_FIELD_NAME
)

# --- OpenSearch Helper Functions ---

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
    except Exception as e:
        print(f"Error during index deletion: {e}")

def index_exists(client, index_name):
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        print(f"Error checking index existence: {e}")
        return False

def create_index_with_knn_mapping(client, index_name, dimension, vector_field):
    """Explicitly defines the index mapping for k-NN and metadata fields."""
    print(f"  - Creating index {index_name} with k-NN mapping...")
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "l2"
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
                "metadata": {
                    "type": "object",
                    "properties": {
                        # Hierarchy Fields
                        "search_term": {"type": "keyword"},
                        "entity_name": {"type": "keyword"},

                        # Distinct Risk Contexts
                        "risk_category": {"type": "keyword"},    # Distinct field
                        "risk_subcategory": {"type": "keyword"}, # Distinct field

                        "priority": {"type": "integer"},
                        "source_id": {"type": "keyword"},
                        "source_name": {"type": "keyword"},
                        "aliases": {"type": "keyword"},
                        "doc_type": {"type": "keyword"}
                    }
                },
                "page_content": {"type": "text"}
            }
        }
    }
    try:
        client.indices.create(index=index_name, body=index_body)
        print(f"  - Successfully created index: {index_name}")
    except opensearch_exceptions.RequestError as e:
        if 'resource_already_exists_exception' in str(e):
            print(f"  - Index {index_name} already exists.")
        else:
            raise

# --- Data Parsing Logic ---

def process_to_review_structure(record: Dict) -> List[Document]:
    """
    Parses the 'risks.toReview' structure.
    Splits Category and Subcategory into distinct context lines.
    """
    docs = []

    # 1. Root Level Info
    search_term = record.get('businessName', 'Unknown')
    root_aliases = record.get('aliases', [])
    if isinstance(root_aliases, list):
        root_aliases = [str(a) for a in root_aliases if a]
    else:
        root_aliases = []

    # 2. Iterate Entities (toReview)
    risks_data = record.get('risks', {})
    to_review_list = risks_data.get('toReview', [])

    for item in to_review_list:
        entity_name = item.get('title', 'Unknown Entity')

        # Iterate Source Records
        for source_rec in item.get('sourceRecord', []):

            # Context details
            entity_aliases = source_rec.get('alias', []) or []
            addresses = source_rec.get('addresses', []) or []
            address_str = ", ".join(addresses) if addresses else "Unknown Address"

            all_aliases = list(set(root_aliases + entity_aliases))
            aliases_str = ", ".join(all_aliases)

            # 3. The Actual Risk Findings
            for finding in source_rec.get('riskFindings', []):

                finding_text = finding.get('findingText', 'N/A')
                category = finding.get('riskCategory', 'N/A')
                subcategory = finding.get('riskSubcategory', 'N/A')
                source_name = finding.get('source', 'Unknown')

                explanations = finding.get('explanations', [])
                explanation_text = " ".join(explanations) if explanations else "N/A"

                source_ids = finding.get('sourceIds', [])
                source_id_val = source_ids[0] if source_ids else "N/A"

                priority = finding.get('priority', 99)

                # Construct Rich Content with SPLIT CONTEXT
                # This layout helps the LLM distinguish between the broad category and specific subcategory
                content = (
                    f"Risk Context for Entity: **{entity_name}**\n"
                    f"(Search Term: {search_term})\n"
                    f"Risk Category: {category}\n"        # Context Line 1
                    f"Risk Subcategory: {subcategory}\n"  # Context Line 2
                    f"Priority Value: {priority}\n"       # <-- ADDED THIS LINE
                    f"Finding Detail: {finding_text}\n"
                    f"Explanation: {explanation_text}\n"
                    f"Source: {source_name}\n"
                    f"Source ID: {source_id_val}\n"
                    f"Address: {address_str}\n"
                    f"Aliases: {aliases_str}"
                )

                metadata = {
                    "search_term": search_term,
                    "entity_name": entity_name,
                    "risk_category": category,       # Trackable Field
                    "risk_subcategory": subcategory, # Trackable Field
                    "priority": int(priority) if priority else 99,
                    "source_id": source_id_val,
                    "source_name": source_name,
                    "aliases": all_aliases,
                    "doc_type": "RiskFinding"
                }

                docs.append(Document(page_content=content, metadata=metadata))

    return docs

def process_file_data(raw_data: Any) -> List[Document]:
    all_docs = []

    if isinstance(raw_data, dict):
        if 'payload' in raw_data:
             records = raw_data['payload'].get('documents', [])
        else:
             records = [raw_data]
    elif isinstance(raw_data, list):
        records = raw_data
    else:
        return []

    for record in records:
        if 'risks' in record and 'toReview' in record['risks']:
            print(f"    - Found 'toReview' structure for {record.get('businessName')}")
            all_docs.extend(process_to_review_structure(record))
        else:
            print(f"    - Skipping record {record.get('businessName', 'Unknown')} (No 'risks.toReview' found)")
            pass

    return all_docs

def _load_and_process_files(file_names, data_dir):
    all_documents = []
    print(f"  - Loading files from: {data_dir}")

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"  - Processing {file_name}...")
            docs = process_file_data(data)
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return all_documents

def ingest_data_to_opensearch(file_names, data_dir, force_reindex=False):
    os_client = get_opensearch_client()

    if force_reindex and index_exists(os_client, INDEX_NAME):
        delete_existing_index(os_client, INDEX_NAME)

    if not index_exists(os_client, INDEX_NAME):
        create_index_with_knn_mapping(os_client, INDEX_NAME, EMBEDDING_DIMENSION, VECTOR_FIELD_NAME)

    docs = _load_and_process_files(file_names, data_dir)
    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        OpenSearchVectorSearch.from_documents(
            docs, embeddings, opensearch_url=OPENSEARCH_URL,
            index_name=INDEX_NAME, client=os_client,
            vector_field=VECTOR_FIELD_NAME,
            client_kwargs={'verify_certs': False, 'ssl_show_warn': False}
        )
        print(f"-> Indexed {len(docs)} documents.")
    else:
        print("-> No documents to index.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    ingest_data_to_opensearch(DATA_FILE_NAMES, DATA_DIR, force_reindex=True)