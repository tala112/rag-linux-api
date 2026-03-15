import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100
MAX_SAMPLES = 5000

DATASETS = [
    "Romit2004/LinuxCommands",
    "bajrangCoder/linux_cmd_alpaca",
]

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collections = qdrant.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Created collection '{COLLECTION_NAME}'")
else:
    qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Recreated collection '{COLLECTION_NAME}'")

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded")

all_data = []
seen_inputs = set()

for ds_name in DATASETS:
    try:
        print(f"Loading {ds_name}...")
        ds = load_dataset(ds_name)
        train_data = ds["train"]
        
        count = 0
        for item in train_data:
            if len(all_data) >= MAX_SAMPLES:
                break
            if isinstance(item, dict):
                inp = None
                out = None
                
                if "cmd" in item and "augmented_text" in item:
                    inp = item.get("augmented_text", "")
                    out = item.get("cmd", "")
                elif "input" in item and "output" in item:
                    inp = item.get("input", "")
                    out = item.get("output", "")
                elif "instruction" in item and "output" in item:
                    inp = item.get("instruction", "")
                    out = item.get("output", "")
                elif "instruction" in item and "response" in item:
                    inp = item.get("instruction", "")
                    out = item.get("response", "")
                
                if inp and out and inp not in seen_inputs:
                    seen_inputs.add(inp)
                    all_data.append({"input": inp, "output": out})
                    count += 1
        
        print(f"  Added {count} items. Total: {len(all_data)}")
    except Exception as e:
        print(f"  Error loading {ds_name}: {e}")

print(f"\nTotal unique commands: {len(all_data)}")

print("Generating embeddings...")
points = []
for idx, item in enumerate(all_data):
    if idx % 500 == 0:
        print(f"  Processing {idx}/{len(all_data)}")
    text = f"Request: {item['input']} Command: {item['output']}"
    embedding = embedder.encode(text).tolist()
    points.append(PointStruct(
        id=idx,
        vector=embedding,
        payload={"input": item["input"], "output": item["output"]}
    ))

print(f"Uploading {len(points)} points...")
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
    print(f"  Uploaded {len(batch)} points (total {i+len(batch)})")

print("Done!")
