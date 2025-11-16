uvicorn src.services.rag_api:app --reload --port 8080	Starts the API Service. Launches the FastAPI application on port 8080 with auto-reloading enabled for development.
uvicorn src.services.rag_api:app --reload --port 8080 --log-level debug	Starts the API in Debug Mode. Runs the server with verbose logging to troubleshoot filter/search failures.
curl -X POST "http://127.0.0.1:8080/ingest/full" -F "file=@data/knowledge-base.json" -F "force_reindex=True"	Triggers Full Data Ingestion. Calls the /ingest/full API endpoint to upload the data file and perform a clean reindex.
curl -X POST "http://127.0.0.1:8080/retrieve" -H "Content-Type: application/json" -d '{"query": "..."}'	Tests the Retrieval Endpoint. Sends a hybrid search query to the /retrieve endpoint to fetch context.
pip install uvicorn	Installs the ASGI Server. Required to run the FastAPI application.
pip install "fastapi[all]"	Installs FastAPI and essential dependencies (like Pydantic, which solved your recent error).