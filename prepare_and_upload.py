import os
import json
import random
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# تحميل المتغيرات
load_dotenv()

# -------------------- الإعدادات --------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100
DATA_FILE = r"archive\complex_linux_commands_million.json"  # استخدمي raw string أو خطوط مائلة أمامية
SAMPLE_SIZE = 10000  # عدد العينات التي نريدها (غيّريها حسب حاجتك)

# -------------------- الاتصال بـ Qdrant --------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# إنشاء المجموعة إذا لم تكن موجودة
collections = qdrant.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created.")
else:
    print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")

# -------------------- تحميل النموذج --------------------
print("⏳ Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Model loaded.")

# -------------------- قراءة البيانات وأخذ عينة --------------------
print(f"📂 Reading data from {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    all_data = json.load(f)  # هذا قد يستهلك ذاكرة ولكن سنأخذ عينة

print(f"✅ Loaded {len(all_data)} records.")

# أخذ عينة عشوائية
sampled_data = random.sample(all_data, min(SAMPLE_SIZE, len(all_data)))
print(f"🎲 Selected {len(sampled_data)} random records for upload.")

# -------------------- رفع العينة إلى Qdrant --------------------
points = []
for idx, item in enumerate(sampled_data):
    text = f"Question: {item['input']} Command: {item['output']}"
    embedding = embedder.encode(text).tolist()

    point = PointStruct(
        id=idx,  # المعرف داخل هذه العينة فقط (يمكنك استخدام hash إذا أردت)
        vector=embedding,
        payload={
            "text": text,
            "input": item["input"],
            "output": item["output"]
        }
    )
    points.append(point)

    if len(points) >= BATCH_SIZE:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"   ✅ Uploaded {len(points)} points (total {idx+1})")
        points = []

if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uploaded final {len(points)} points.")

print("🎉 Sample data uploaded successfully!")
'''
import os
import json
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# تحميل المتغيرات من .env
load_dotenv()

# -------------------- الإعدادات --------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # حجم المتجه 384
BATCH_SIZE = 100  # رفع 100 نقطة كل مرة
DATA_FILE = "archive\\complex_linux_commands_million.json"  # اسم ملف البيانات

# -------------------- الاتصال بـ Qdrant --------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# التأكد من وجود المجموعة (collection)
collections = qdrant.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created.")
else:
    print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists.")

# -------------------- تحميل نموذج التضمين (embedding) --------------------
print("⏳ Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Model loaded.")

# -------------------- قراءة البيانات --------------------
print(f"📂 Reading data from {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)  # هذا قد يستهلك ذاكرة كبيرة إذا كان الملف ضخماً
print(f"✅ Loaded {len(data)} records.")

# -------------------- تجهيز النقاط ورفعها --------------------
points = []
for idx, item in enumerate(data):
    # إنشاء نص وصفي (يمكنك تغيير الصيغة)
    text = f"Question: {item['input']} Command: {item['output']}"

    # توليد embedding
    embedding = embedder.encode(text).tolist()

    # إنشاء نقطة (point) في Qdrant
    point = PointStruct(
        id=idx,  # معرف فريد (يمكنك استخدام أي رقم)
        vector=embedding,
        payload={
            "text": text,
            "input": item["input"],
            "output": item["output"]
        }
    )
    points.append(point)

    # رفع الدفعة إذا اكتملت
    if len(points) >= BATCH_SIZE:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"   ✅ Uploaded {len(points)} points (total {idx+1})")
        points = []

# رفع أي نقاط متبقية
if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uploaded final {len(points)} points.")

print("🎉 All data uploaded to Qdrant!")
'''