Setting Up OpenSearch for RAG (Docker Desktop)

Since you have Docker Desktop, setting up OpenSearch is straightforward. You will run a single-node, un-clustered instance and map its port so your Python code can access it directly from your host machine.

Step 1: Run the OpenSearch Container

Open your terminal (PowerShell, Command Prompt, or VS Code terminal) and run the following command.
This pulls the OpenSearch image and starts the container with security enabled (default behavior).

docker run -d \
  --name opensearch-rag-poc \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=opensearch" \
  opensearchproject/opensearch:2.11.1

Parameter Reference
Parameter	Purpose
-d	Runs the container in detached mode (background).
--name	Assigns a memorable name to the instance.
-p 9200:9200	Maps the container port 9200 to your host machine’s port 9200. This is the connection point for Python.
-e "discovery.type=single-node"	Required for running a single-node instance for development.
-e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=..."	Sets the initial password for the default user admin. Crucial for connecting.
Step 2: Verify Connection and Credentials
✅ Check Status

Open Docker Desktop and confirm the container named opensearch-rag-poc is running and healthy.

🌐 Verify Access (Browser)

Navigate to:
👉 https://localhost:9200

⚠️ You may receive a browser security warning because OpenSearch uses self-signed SSL certificates by default.
Simply bypass the warning to proceed.

Credentials:

Username: admin

Password: opensearch

🧩 Confirm Health (API)

Once authenticated, run a simple check in your terminal:

curl -k -u admin:opensearch https://localhost:9200/_cat/health?v


The -k flag bypasses the self-signed SSL certificate check.
You should see a status of green or yellow.

Step 3: Python Connection Details

Now that OpenSearch is running, your Python application (using either opensearch-py or the LangChain OpenSearchVectorSearch library) will connect using the following parameters:

Parameter	Value	Notes
Host/URL	https://localhost	Use HTTPS due to default security settings.
Port	9200	The mapped port.
Username	admin	Default user.
Password	myOpenSearchPass	The value set in Step 1.
SSL Flag	verify_certs=False	Required to ignore local self-signed certificate warnings.

✅ Your OpenSearch instance is now ready to receive:

Index creation scripts

Vectorized data from your Python code

Next Steps:

Use opensearch-py or LangChain’s vector store to create and populate an index.

Store embeddings for retrieval-augmented generation (RAG) workflows.


OpenSearch with no secure access:

docker run -d \
  --name opensearch-insecure \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "OPENSEARCH_INITIAL_SECURITY_DEMO_ADMIN_PASSWORD=opensearch" \
  -e "plugins.security.disabled=true" \
  opensearchproject/opensearch:2.11.1