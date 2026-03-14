import os
import json
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100
DATA_FILE = r"archive\complex_linux_commands_million.json"
SAMPLE_SIZE = 10000

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collections = qdrant.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Created collection '{COLLECTION_NAME}'")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists")

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded")

print(f"Reading data from {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    all_data = json.load(f)

print(f"Loaded {len(all_data)} records")

sampled_data = random.sample(all_data, min(SAMPLE_SIZE, len(all_data)))
print(f"Selected {len(sampled_data)} random records")

total_uploaded = 0
for batch_start in range(0, len(sampled_data), BATCH_SIZE):
    batch = sampled_data[batch_start:batch_start + BATCH_SIZE]
    points = []
    
    for idx, item in enumerate(batch):
        global_idx = batch_start + idx
        text = f"Request: {item['input']} Command: {item['output']}"
        embedding = embedder.encode(text).tolist()

        point = PointStruct(
            id=global_idx,
            vector=embedding,
            payload={
                "input": item["input"],
                "output": item["output"]
            }
        )
        points.append(point)

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    total_uploaded += len(points)
    print(f"Uploaded {len(points)} points (total {total_uploaded})")

print("Done!")
