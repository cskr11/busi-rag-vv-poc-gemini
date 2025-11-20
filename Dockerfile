# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for build tools)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements (you can create a requirements.txt or use this inline list)
# We install the specific libraries used in your code
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    langchain \
    langchain-community \
    langchain-google-genai \
    opensearch-py \
    pydantic \
    python-dotenv \
    httpx

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Command to run the API (Note: points to services.rag_api based on your folder structure)
CMD ["uvicorn", "services.rag_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]