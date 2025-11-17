#ingestion/ingest.py
import os
import sys
import json
from collections import defaultdict

# NOTE: Since the full project structure isn't available, 
# assuming necessary components like config are either defined 
# or handled in your actual environment.
# Setting paths relative to the current working directory for file access.

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
from langchain_core.documents import Document

# --- CONSTANTS (Adjust these to match your OpenSearch setup) ---
# Ensure the parent directory (src) is in the path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, exceptions as opensearch_exceptions
# Document is now correctly aliased from langchain_core.documents
from langchain_core.documents import Document
from config.config import DATA_DIR, DATA_FILE_NAMES, OPENSEARCH_URL, INDEX_NAME, EMBEDDING_MODEL, EMBEDDING_DIMENSION, VECTOR_FIELD_NAME
# --- Helper Functions (Standard OpenSearch management functions) ---

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
    """Explicitly defines the index mapping for k-NN and metadata fields."""
    print(f"  - Creating index {index_name} with k-NN mapping (dimension: {dimension})...")
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
                "metadata": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "keyword"},
                        "risk_category": {"type": "keyword"},
                        "risk_subcategory": {"type": "keyword"},
                        "priority": {"type": "integer"},
                        "doc_type": {"type": "keyword"}, # 'RiskFinding' or 'ProfileSummary'
                        "profile_score": {"type": "float"}, 
                        "total_findings": {"type": "integer"}, 
                        "risk_categories_summary": {"type": "text"},
                        "primary_address": {"type": "text"},
                        "duns_id": {"type": "keyword"},
                        "source_name": {"type": "keyword"},
                        "source_id": {"type": "keyword"}
                    }
                },
                "page_content": {"type": "text"}
            }
        }
    }
    print(f"  - Index mapping defined. Creating index... {index_body}")
    try:
        client.indices.create(index=index_name, body=index_body)
        print(f"  - Successfully created index: {index_name}")
    except opensearch_exceptions.RequestError as e:
        if 'resource_already_exists_exception' in str(e):
            print(f"  - Index {index_name} already exists. Skipping creation.")
        else:
            print(f"Error creating index: {e}")
            raise

# ----------------------------------------------------------------------
# --- CORE DATA PROCESSING LOGIC ---
# ----------------------------------------------------------------------

def extract_business_info(document):
    """Extracts primary business name, address, and DUNS ID."""
    identity_doc = document.get('sourceDocument', {}).get('identitySourceDocument', {})
    
    # 1. Extract primary business name (Prioritize type 2, then any name)
    business_name = "N/A"
    name_identifiers = identity_doc.get('businessNameIdentifiers', [])
    
    primary_name_candidates = [
        ni['identifier']['data']['value']
        for ni in name_identifiers 
        if ni.get('identifier', {}).get('data', {}).get('type') == 2
    ]
    if primary_name_candidates:
        business_name = primary_name_candidates[0]
    elif name_identifiers:
        # Fallback to the first found name
        business_name = name_identifiers[0]['identifier']['data']['value']

    # 2. Extract a concise address
    primary_address = 'N/A'
    address_identifiers = identity_doc.get('addressIdentifiers', [])
    
    # Try to find a 'Primary' address type
    primary_address_candidates = [
        ai for ai in address_identifiers 
        if ai.get('identifier', {}).get('type') == 'Primary'
    ]
    
    address_to_use = primary_address_candidates[0] if primary_address_candidates else address_identifiers[0] if address_identifiers else None

    if address_to_use:
        # Concatenate address lines, city, state, and country
        data = address_to_use.get('identifier', {}).get('data', {})
        parts = [
            data.get('line1', ''), 
            data.get('line2', ''),
            data.get('line3', ''),
            data.get('city', ''), 
            data.get('state', ''), 
            data.get('country', '')
        ]
        # Filter empty strings and clean up path separators
        primary_address = ", ".join(filter(None, parts)).strip().replace('\\', ' ')

        
    # 3. Extract DUNS ID
    duns_id = "N/A"
    duns_identifiers = identity_doc.get('dunsIdentifiers', [])
    if duns_identifiers:
        duns_id = duns_identifiers[0]['identifier']['data']['number']
        
    return business_name, primary_address, duns_id

def process_risk_data_to_documents(raw_data):
    """
    Parses business risk profiles to generate distinct Document objects:
    one for every specific risk finding, and one summary for the profile.
    """
    document_list = []
    
    for risk_profile in raw_data.get('payload', {}).get('documents', []):
        doc_id = risk_profile.get('documentId')
        
        # Extract base information for this document
        primary_business_name, business_address, duns_id = extract_business_info(risk_profile)
        
        source_map = risk_profile.get('sourceMap', {})
        risk_findings = risk_profile.get('riskFindings', [])
        risk_categories_present = set()

        # 1. Generate Risk Finding Documents (Granular for semantic search)
        for finding in risk_findings:
            risk_category = finding.get('riskCategory', 'N/A')
            # Strip whitespace in case of data inconsistencies like the H1B example
            risk_subcategory = finding.get('riskSubcategory', 'N/A').strip() 
            risk_priority = finding.get('priority', 99) 
            finding_text = finding.get('findingText', 'N/A')
            
            risk_categories_present.add(risk_category)

            explanations = finding.get('explanations', [])
            explanation_sources = "; ".join(explanations).replace('\\', ' ')
            
            source_ids = finding.get('sourceIds', [])
            
            # Create a separate document for each source ID linked to this finding
            for source_id in source_ids:
                source_name = source_map.get(source_id, {}).get('sourceName', 'Unknown Source')
                
                # Content for vectorization: ensures the model has full context of the risk
                content_text = (
                    f"The business **{primary_business_name}** (DUNS: {duns_id}) located at {business_address} "
                    f"has a security risk. Classification: **{risk_category}** / **{risk_subcategory}**. "
                    f"Priority: {risk_priority}. Finding: {finding_text}. Source: {source_name}. "
                    f"Details: {explanation_sources}"
                )
                
                # Metadata for filtering
                metadata = {
                    "business_id": doc_id, 
                    "company_name": primary_business_name,
                    "doc_type": "RiskFinding",
                    "risk_category": risk_category, 
                    "risk_subcategory": risk_subcategory,
                    "priority": risk_priority, 
                    "primary_address": business_address,
                    "duns_id": duns_id,
                    "source_name": source_name,
                    "source_id": source_id
                }

                document_list.append(Document(page_content=content_text, metadata=metadata))

        # 2. Generate the Profile Summary Document (High-level for overview queries)
        profile_score = risk_profile.get('profileScoreDetail', {}).get('profileScore', 0.0)
        
        summary_content = (
            f"Overall risk profile summary for the business **{primary_business_name}** "
            f"located at {business_address}. It has an overall score of {profile_score}. "
            f"Total detailed risk findings: {len(risk_findings)}. "
            f"Risk categories present: {', '.join(sorted(list(risk_categories_present))) if risk_categories_present else 'None'}."
        )
        
        summary_metadata = {
            "business_id": doc_id, 
            "company_name": primary_business_name,
            "doc_type": "ProfileSummary", 
            "profile_score": profile_score, 
            "total_findings": len(risk_findings),
            "risk_categories_summary": ', '.join(sorted(list(risk_categories_present))),
            "primary_address": business_address,
            "duns_id": duns_id
        }
        document_list.append(Document(page_content=summary_content, metadata=summary_metadata))

    return document_list


# --- Core Ingestion Logic ---

def _load_and_process_files(file_names, data_dir):
    """Loads and processes multiple JSON files from a specified directory."""
    all_documents = []
    print(f"  - Loading files from directory: {data_dir}")
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
             print(f"Warning: File not found at {file_path}. Skipping.")
             continue
             
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            print(f"  - Processing data from {file_name}...")
            # Extend the main list with documents from the current file
            all_documents.extend(process_risk_data_to_documents(raw_data))

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: File not found or is not valid JSON at {file_path}: {e}")
        except Exception as e:
            print(f"An error occurred during data processing for {file_path}: {e}")
            
    print(f"  - Finished processing all files. Total documents generated: {len(all_documents)}")
    return all_documents

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

def ingest_data_to_opensearch(file_names, data_dir, force_reindex=False):
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
    documents_to_ingest = _load_and_process_files(file_names, data_dir)
    if not documents_to_ingest: return
    print(f"  - Total documents to ingest: {len(documents_to_ingest)}")
    print(f"  - Documents to ingest: {documents_to_ingest}")
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, request_options={"timeout": 60})
    print(f"  - Initialized Google Embeddings model: {EMBEDDING_MODEL}")
    
    _run_opensearch_ingestion(documents_to_ingest, os_client, embeddings)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create the dummy data/ directory for demonstration if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Write the uploaded JSON data to the new paths
    # Note: In a real environment, you'd load from the files directly or use the original upload mechanism.
    # For this simulated environment, we recreate the files based on the content provided.
    
    # IBM Data (Simplified to avoid recreating the massive JSON content)
    ibm_data = {
        "successful": True,
        "payload": {
            "documents": [
                {
                    "documentId": "295c9afd-9e14-4502-b94a-580dfb5d5cd3",
                    "sourceDocument": {
                        "identitySourceDocument": {
                            "addressIdentifiers": [{"identifier": {"type": "Primary", "data": {"line1": "1 New Orchard Rd Ste 1", "city": "Armonk", "state": "New York", "country": "United States"}}}],
                            "businessNameIdentifiers": [{"identifier": {"data": {"value": "International Business Machines Corporation", "type": 2}}}]
                        }
                    },
                    "riskFindings": [
                        {"findingText": "Company is listed as a registered entity in SAM.gov.", "riskCategory": "Subversion, Exploitation, Espionage", "riskSubcategory": "US SAM.gov registrant, public discoverability", "priority": 4, "explanations": ["..."], "sourceIds": ["KXAiMjxEZJn7Mxj_5RNXkENBzm5Y="]},
                        {"findingText": "Organization is listed as being an Autonomous System Number (ASN) holder by Hurricane Electric...", "riskCategory": "Foreign Ownership, Control, or Influence (FOCI)", "riskSubcategory": "Cyber Espionage", "priority": 4, "explanations": ["..."], "sourceIds": ["KhYp93lySq3XapIYZikcKkUX8UKE="]}
                    ],
                    "sourceMap": {"KXAiMjxEZJn7Mxj_5RNXkENBzm5Y=": {"sourceName": "BSD"}, "KhYp93lySq3XapIYZikcKkUX8UKE=": {"sourceName": "BSD"}},
                    "profileScoreDetail": {"profileScore": 0.25}
                }
            ]
        }
    }
    
    # Myntra Data (Simplified)
    myntra_data = {
        "successful": True,
        "payload": {
            "documents": [
                {
                    "documentId": "23518eef-5cfb-4014-a3c0-410a49eb3eb0",
                    "sourceDocument": {
                        "identitySourceDocument": {
                            "addressIdentifiers": [{"identifier": {"type": "Primary", "data": {"line1": "Buildings Alyssa, Begonia & Clover, Embassy Tech Village,", "line2": "Outer Ring Road, Devarabeesanahalli Village", "city": "Bengaluru", "state": "Karnataka", "country": "India"}}}],
                            "businessNameIdentifiers": [{"identifier": {"data": {"value": "MYNTRA JABONG INDIA PRIVATE LIMITED", "type": 2}}}]
                        }
                    },
                    "riskFindings": [
                        {"findingText": "Company is listed as a local business in a Hong Kong business registry.", "riskCategory": "Foreign Ownership, Control, or Influence (FOCI)", "riskSubcategory": "ADVERSARIAL SUPPLY CHAIN: Operates in an adversary-controlled location", "priority": 3, "explanations": ["..."], "sourceIds": ["KJg4LodEhzcuN25xIQ0_gpdc3L18="]}
                    ],
                    "sourceMap": {"KJg4LodEhzcuN25xIQ0_gpdc3L18=": {"sourceName": "BSD"}},
                    "profileScoreDetail": {"profileScore": 0.5}
                }
            ]
        }
    }

    with open(os.path.join(DATA_DIR, 'ibm.json'), 'w') as f:
        json.dump(ibm_data, f)
    with open(os.path.join(DATA_DIR, 'myntra.json'), 'w') as f:
        json.dump(myntra_data, f)

    print("\n" + "="*50)
    print("MODE: FULL REINDEX (Deletes and recreates index for all files)")
    # Note: Running this will fail if OpenSearch is not available or if the Gemini API key is missing/invalid.
    # The primary goal here is to demonstrate the correct data structuring logic.
    ingest_data_to_opensearch(DATA_FILE_NAMES, DATA_DIR, force_reindex=True)
    print("="*50)