from opensearchpy import OpenSearch, RequestsHttpConnection, exceptions

HOST = 'localhost'
PORT = 9200
#AUTH = ('admin', 'opensearch') # Adjust to your password
INDEX_NAME = 'weather'

# --- 1. Client Initialization (Secure Connection Fix) ---
# Ensure your client handles the HTTPS and self-signed cert from Docker
client = OpenSearch(
    hosts=[{'host': HOST, 'port': PORT}],
    #http_auth=AUTH,
    use_ssl=False,
    verify_certs=False,
    connection_class=RequestsHttpConnection
)

# --- 2. Sample Documents ---
weather_data = [
    # 1. Marlboro - Your original entry
    {
        "city": "Marlboro",
        "date": "2025-11-11T14:00:00Z",
        "temperature_c": 18,
        "condition": "Sunny and clear",
        "wind_speed_kph": 5,
        "text": "The weather in Marlboro on November 11th is 18°C. Conditions are perfectly sunny and clear with a very light wind speed of 5 kph. Travel conditions are excellent."
    },
    # 2. New York - Original entry, slightly expanded
    {
        "city": "New York",
        "date": "2025-11-11T14:00:00Z",
        "temperature_c": 16,
        "condition": "Partly cloudy",
        "wind_speed_kph": 15,
        "text": "In New York City, it is partly cloudy with a temperature of 16°C. The wind speed is moderate at 15 kph, which could lead to minor delays at regional airports due to crosswinds."
    },
    # 3. Los Angeles - Original entry, slightly expanded
    {
        "city": "Los Angeles",
        "date": "2025-11-11T14:00:00Z",
        "temperature_c": 24,
        "condition": "Foggy morning, clear afternoon",
        "wind_speed_kph": 8,
        "text": "Los Angeles is experiencing a high temperature of 24°C. Note that the morning was impacted by a dense fog advisory, which has since lifted, clearing the way for a sunny afternoon."
    },
    # --- New Data for Context/Retrieval Testing ---
    
    # 4. Miami - High-risk (Storm) scenario
    {
        "city": "Miami",
        "date": "2025-11-11T14:00:00Z",
        "temperature_c": 28,
        "condition": "Heavy thunderstorms",
        "wind_speed_kph": 45,
        "text": "Miami is under a severe weather alert. Heavy thunderstorms are expected, and temperatures are 28°C. Wind speeds are high at 45 kph. Flights and marine activities are strongly advised to cease immediately. This represents a **high-impact risk event**."
    },
    # 5. Chicago - Temperature query test
    {
        "city": "Chicago",
        "date": "2025-11-12T10:00:00Z", # A different day/time for better filtering tests later
        "temperature_c": 4,
        "condition": "Overcast and Cold",
        "wind_speed_kph": 20,
        "text": "Chicago on November 12th is extremely cold, with an overcast sky and temperature hovering around 4°C. The wind chill factor makes the 'feels like' temperature near zero, demanding winter clothing."
    },
    # 6. San Diego - Similar but different city to Los Angeles
    {
        "city": "San Diego",
        "date": "2025-11-11T14:00:00Z",
        "temperature_c": 22,
        "condition": "Mild and Coastal Fog",
        "wind_speed_kph": 10,
        "text": "San Diego reports mild weather at 22°C, but a persistent low-lying coastal fog remains, impacting visibility on major freeways and delaying some ferry services in the harbor area. Be cautious when driving near the coast."
    }
]
# --- 3. Indexing Documents ---
# print(f"Starting ingestion into index: {INDEX_NAME}")

# success_count = 0
# error_count = 0

# for i, doc in enumerate(weather_data):
#     try:
#         response = client.index(index=INDEX_NAME, body=doc, id=i + 1)
#         print(f"Indexed document ID {i + 1}: {response['result']}")
#         success_count += 1
#     except exceptions.TransportError as e:
#         print(f"Error indexing document {i + 1}: {e}")
#         error_count += 1
#         # If the index does not exist, OpenSearch will create it here.

# # --- 4. Verify Count ---
# try:
#     count = client.count(index=INDEX_NAME)['count']
#     print(f"\nSuccessfully indexed {success_count} documents into the '{INDEX_NAME}' index.")
#     if error_count > 0:
#         print(f"{error_count} documents failed to index.")
#     print(f"Total documents in index (may include previous data): {count}")
# except exceptions.NotFoundError:
#     print(f"\nIndex '{INDEX_NAME}' was not found.")
# retrieve and print all documents to verify
try:
    response = client.search(index=INDEX_NAME, body={"query": {"match_all": {}}})
    print("\nRetrieved documents:")
    for hit in response['hits']['hits']:
        print(f"ID: {hit['_id']}, Source: {hit['_source']}")
except exceptions.NotFoundError:
    print(f"\nIndex '{INDEX_NAME}' was not found.")
