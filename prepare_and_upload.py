import os
import json
import uuid
import hashlib

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

BATCH_SIZE = 32
CHUNK_SIZE = 100
TOP_K = 5

PROGRESS_FILE = "progress.json"


DATASETS = [
    "hotal/linux_commands",
    "rohanbalkondekar/linux_commands",
    "Romit2004/LinuxCommands",
]


def should_filter_command(cmd):
    if not cmd:
        return True
    if len(cmd) > 50:
        return True
    ''' if "|" in cmd:
        return True
    if "$(" in cmd:
        return True
    if "&&" in cmd or "||" in cmd:
        return True
    if ";" in cmd:
        return True
    if cmd.startswith("if ") or cmd.startswith("for ") or cmd.startswith("while "):
        return True
    if cmd.count(" ") > 4:
        return True'''
    return False


if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
else:
    progress = {ds: 0 for ds in DATASETS}


def save_progress():
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collections = qdrant.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print("Collection created")
else:
    print("Collection exists")
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        info_dict = info.dict() if hasattr(info, 'dict') else {}
        vectors_config = info_dict.get("params", {}).get("vectors", {})
        if vectors_config:
            existing_size = vectors_config.get("size")
            if existing_size and existing_size != 768:
                print(f"Deleting collection (size mismatch: {existing_size} != 768)")
                qdrant.delete_collection(COLLECTION_NAME)
                qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                progress = {ds: 0 for ds in DATASETS}
                save_progress()
                print("Collection recreated")
            else:
                print(f"Collection exists with size {existing_size or 'unknown'}")
    except Exception as e:
        print(f"Could not check collection: {e}")


print("Loading model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Model loaded")


def pair_hash(inp, out):
    text = inp.strip() + "||" + out.strip()
    return hashlib.sha256(text.encode()).hexdigest()


existing_hashes = set()
print("Reading existing vectors...")
try:
    offset = 0
    limit = 500
    while True:
        res = qdrant.scroll(collection_name=COLLECTION_NAME, limit=limit, offset=offset)
        points = res.points
        if not points:
            break
        for p in points:
            inp = p.payload.get("input", "")
            out = p.payload.get("output", "")
            existing_hashes.add(pair_hash(inp, out))
        offset += len(points)
except Exception as e:
    print("Error reading:", e)
print("Existing:", len(existing_hashes))


def parse_item(item, ds_name):
    inp, out = None, None

    if ds_name == "hotal/linux_commands":
        cmd = item.get("command")
        resp = item.get("response")
        if cmd and resp:
            inp = resp
            out = cmd

    elif ds_name == "rohanbalkondekar/linux_commands":
        cmd = item.get("input")
        text = item.get("output")
        if cmd and text:
            inp = text
            out = cmd

    elif ds_name == "Romit2004/LinuxCommands":
        cmd = item.get("cmd")
        text = item.get("augmented_text") or item.get("invocation")
        if cmd and text:
            inp = text
            out = cmd

    return inp, out


def encode_and_upload(data_chunk):
    if not data_chunk:
        return 0

    texts = [x["input"] for x in data_chunk]

    print(f"  Encoding {len(texts)} items...")
    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"input": item["input"], "output": item["output"]})
        for item, emb in zip(data_chunk, embeddings)
    ]

    print(f"  Uploading {len(points)} points...")
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"    Uploaded {i + len(batch)}/{len(points)}")

    return len(points)


total_uploaded = 0
total_filtered = 0

for ds_name in DATASETS:
    ds_start = progress.get(ds_name, 0)

    print(f"\n{'='*50}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*50}")

    ds = load_dataset(ds_name)
    train = ds["train"]
    total = len(train)
    print(f"Total items: {total}")

    if ds_start >= total:
        print("Already completed, skipping")
        continue

    while ds_start < total:
        chunk_end = min(ds_start + CHUNK_SIZE, total)
        print(f"\nChunk: {ds_start} -> {chunk_end}")

        chunk_data = []
        for i in range(ds_start, chunk_end):
            item = train[i]
            inp, out = parse_item(item, ds_name)
            
            if not inp or not out:
                continue
            
            if should_filter_command(out):
                total_filtered += 1
                continue
            
            h = pair_hash(inp, out)
            if h in existing_hashes:
                continue
            
            existing_hashes.add(h)
            chunk_data.append({"input": inp, "output": out})

        if chunk_data:
            count = encode_and_upload(chunk_data)
            total_uploaded += count
            print(f"  Added: {count} new items")
        else:
            print("  No new items in this chunk")

        progress[ds_name] = chunk_end
        save_progress()
        print(f"  Progress saved: {chunk_end}/{total}")

        ds_start = chunk_end

        print("  Clearing memory...")
        del chunk_data

    print(f"Dataset '{ds_name}' completed!")


print(f"\n{'='*50}")
print(f"DONE!")
print(f"Total uploaded: {total_uploaded}")
print(f"Filtered (complex commands): {total_filtered}")
print(f"{'='*50}")
