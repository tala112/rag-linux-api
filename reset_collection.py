import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "linux_commands"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collections = qdrant.get_collections().collections
if COLLECTION_NAME in [c.name for c in collections]:
    qdrant.delete_collection(COLLECTION_NAME)
    print(f"Deleted collection '{COLLECTION_NAME}'")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist")
