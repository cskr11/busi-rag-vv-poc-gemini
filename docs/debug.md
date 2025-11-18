uvicorn src.services.rag_api:app --reload --port 8080	Starts the API Service. Launches the FastAPI application on port 8080 with auto-reloading enabled for development.Ok,
uvicorn src.services.rag_api:app --reload --port 8080 --log-level debug	Starts the API in Debug Mode. Runs the server with verbose logging to troubleshoot filter/search failures.

curl command to ingest files:

curl -X POST "http://localhost:8080/ingest/configured_full" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "force_reindex=true"

curl -X POST "http://localhost:8080/ingest/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@new_vendor_data.json;type=application/json" \
     -F "force_reindex=false"


curl -X POST "http://127.0.0.1:8080/retrieve" -H "Content-Type: application/json" -d '{"query": "..."}'	Tests the Retrieval Endpoint. Sends a hybrid search query to the /retrieve endpoint to fetch context.
pip install uvicorn	Installs the ASGI Server. Required to run the FastAPI application.
pip install "fastapi[all]"	Installs FastAPI and essential dependencies (like Pydantic, which solved your recent error).

1. What are the risks associated with a company? (e.g., IBM)
   1. Semantic Search the company name. 2. Filter by file_source_tag (or company_name) and ensure doc_type is 'RiskFinding'.
   2. {"term": {"metadata.file_source_tag": "ibm"}, "term": {"metadata.doc_type": "RiskFinding"}}
2. Which companies are having the high priority risk?
   1. Semantic Search for keywords like "highest risk" or "critical findings." 
   2. Filter by a priority range, since lower numbers often mean higher risk (e.g., Priority <= 5).
   3. {"range": {"metadata.priority": {"lte": 5}}, "term": {"metadata.doc_type": "RiskFinding"}}
3. The companies that are not having any risks.
   1. Semantic Search for a general query ("company profile summary"). 2. Filter by doc_type is 'ProfileSummary' AND total_findings is 0
   2. {"term": {"metadata.doc_type": "ProfileSummary"}, "term": {"metadata.total_findings": 0}}
4. Drill down to a risk. (e.g., for Myntra, about 'Sanctions')
   1. Semantic Search the specific risk category. 2. Filter by file_source_tag (or company name) AND risk_category.
   2. {"term": {"metadata.file_source_tag": "myntra"}, "term": {"metadata.risk_category": "Sanctions"}}
5. Find the relevant docs for the risk & company (e.g., IBM and Contested Industry)
   1. Semantic Search the finding text. 2. Filter by file_source_tag AND risk_category. This is the most precise query.
   2. {"term": {"metadata.file_source_tag": "ibm"}, "term": {"metadata.risk_category": "Government Restrictions"}}
