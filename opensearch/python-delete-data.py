from opensearchpy import OpenSearch, RequestsHttpConnection

# --- Client Initialization (Based on your input) ---
HOST = 'localhost' # Replace with your host
PORT = 9200        # Replace with your port
INDEX_NAME = 'weather' # **Replace with the index you want to clear**

client = OpenSearch(
    hosts=[{'host': HOST, 'port': PORT}],
    # http_auth=('user', 'password'), # Uncomment and replace if you use basic auth
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection
)

# --- The Deletion Request ---
try:
    response = client.delete_by_query(
        index=INDEX_NAME,
        body={
            "query": {
                "match_all": {}
            }
        }
    )
    
    # Print the response details
    print(f"Successfully initiated 'delete_by_query' for index: {INDEX_NAME}")
    print(f"Documents deleted: {response.get('deleted')}")
    print("Response details:")
    print(response)

except Exception as e:
    print(f"An error occurred during delete_by_query: {e}")