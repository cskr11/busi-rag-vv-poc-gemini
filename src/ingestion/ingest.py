import os
import sys
import json
from collections import defaultdict
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
                        "business_id": {"type": "keyword"},
                        "company_name": {"type": "keyword"},
                        "doc_type": {"type": "keyword"},
                        "risk_category": {"type": "keyword"},
                        "risk_subcategory": {"type": "keyword"},
                        "priority": {"type": "integer"},
                        "profile_score": {"type": "float"},
                        "total_findings": {"type": "integer"},
                        "primary_address": {"type": "text"},
                        "duns_id": {"type": "keyword"},
                        "source_name": {"type": "keyword"},
                        "aliases": {"type": "keyword"}, # Added aliases field
                        "source_id": {"type": "keyword"}
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

# --- Data Processing Logic ---

def extract_business_info(document):
    """Robustly extracts core identity attributes."""
    source_doc = document.get('sourceDocument', {})
    identity_doc = source_doc.get('identitySourceDocument', {})

    # 1. Extract Business Name
    business_name = "N/A"
    name_identifiers = identity_doc.get('businessNameIdentifiers', [])
    candidates = [n['identifier']['data']['value'] for n in name_identifiers
                  if n.get('identifier', {}).get('data', {}).get('type') == 2]
    if candidates:
        business_name = candidates[0]
    elif name_identifiers:
         business_name = name_identifiers[0]['identifier']['data'].get('value', 'N/A')

    # Fallback: Check if businessName is at the root
    if (business_name == "N/A" or not business_name) and 'businessName' in document:
         business_name = document.get('businessName')

    # 2. Extract Address
    primary_address = "N/A"
    address_identifiers = identity_doc.get('addressIdentifiers', [])
    addr_candidates = [a for a in address_identifiers if a.get('identifier', {}).get('data', {}).get('type') == 'Primary']
    addr_obj = addr_candidates[0] if addr_candidates else (address_identifiers[0] if address_identifiers else None)

    if addr_obj:
        primary_address = addr_obj.get('identifier', {}).get('displayValue', 'N/A')

    # 3. Extract DUNS
    duns_id = "N/A"
    duns_identifiers = identity_doc.get('dunsIdentifiers', [])
    if duns_identifiers:
        duns_id = duns_identifiers[0]['identifier']['data'].get('number', 'N/A')

    return str(business_name).strip(), str(primary_address).strip(), str(duns_id).strip()

def process_risk_data_to_documents(raw_data: Dict) -> List[Document]:
    """
    Parses JSON to generate rich Document objects.
    Handles 'aliases' extraction and nested 'riskSearchDocuments' structure.
    """
    document_list = []

    # Handle structure variations
    if isinstance(raw_data, list):
        docs_array = raw_data
    else:
        docs_array = raw_data.get('payload', {}).get('documents', []) if 'payload' in raw_data else []

    for record in docs_array:
        # --- EXTRACT ALIASES (from root record) ---
        # Normalize aliases to a list of strings
        raw_aliases = record.get('aliases', [])
        if isinstance(raw_aliases, list):
             aliases = [str(a) for a in raw_aliases if a]
        else:
             aliases = []

        # --- UNWRAP LOGIC ---
        if 'riskSearchDocuments' in record:
            risk_profile = record['riskSearchDocuments']
            wrapper_name = record.get('businessName') 
        else:
            risk_profile = record
            wrapper_name = None

        doc_id = risk_profile.get('documentId', 'Unknown')
        
        comp_name, address, duns = extract_business_info(risk_profile)
        
        if (comp_name == "N/A" or comp_name == "None") and wrapper_name:
            comp_name = wrapper_name

        source_doc = risk_profile.get('sourceDocument', {})
        risk_findings = source_doc.get('riskFindings', [])
        source_details_map = source_doc.get('sourceDetailsBySourceKey', {})
        profile_score = source_doc.get('profileScoreDetail', {}).get('profileScore', 0.0)

        aliases_str = ", ".join(aliases) if aliases else "None"

        # 1. Create Granular Risk Documents
        for finding in risk_findings:
            category = str(finding.get('riskCategory') or 'N/A')
            subcategory = str(finding.get('riskSubcategory') or 'N/A')
            priority = finding.get('priority', 99)
            text = str(finding.get('findingText') or 'N/A')

            src_ids = finding.get('sourceIds', [])
            src_name = "Unknown"
            source_id_value = "N/A"
            
            if src_ids:
                source_id_value = src_ids[0]
                if src_ids[0] in source_details_map:
                    src_name = source_details_map[src_ids[0]].get('sourceName', 'Unknown')
            elif finding.get('id'):
                source_id_value = finding.get('id')

            # Construct Page Content (Include Aliases for Keyword Search)
            content = (
                f"Risk Finding for **{comp_name}**.\n"
                f"Aliases: {aliases_str}\n"
                f"Category: {category} / {subcategory}\n"
                f"Priority: {priority}\n"
                f"Finding: {text}\n"
                f"Source: {src_name}\n"
                f"Source ID: {source_id_value}\n"
                f"DUNS: {duns}"
            )

            metadata = {
                "company_name": comp_name,
                "aliases": aliases, # Store as list for filtering/matching
                "risk_category": category,
                "risk_subcategory": subcategory,
                "priority": priority,
                "doc_type": "RiskFinding",
                "source_name": src_name,
                "duns_id": duns,
                "business_id": doc_id,
                "source_id": source_id_value
            }
            document_list.append(Document(page_content=content, metadata=metadata))

        # 2. Create Profile Summary Document
        summary_content = (
            f"Risk Profile Summary for **{comp_name}**.\n"
            f"Aliases: {aliases_str}\n"
            f"Overall Score: {profile_score}\n"
            f"Total Findings: {len(risk_findings)}\n"
            f"Address: {address}"
        )

        summary_metadata = {
            "company_name": comp_name,
            "aliases": aliases,
            "doc_type": "ProfileSummary",
            "profile_score": profile_score,
            "total_findings": len(risk_findings),
            "duns_id": duns,
            "business_id": doc_id,
            "source_id": "N/A"
        }

        print(f"-> Processed profile for {comp_name} with content: {summary_content} metadata: {summary_metadata}.")

        document_list.append(Document(page_content=summary_content, metadata=summary_metadata))

    return document_list

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
            # Passed data directly, file_source_tag removed
            docs = process_risk_data_to_documents(data)
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