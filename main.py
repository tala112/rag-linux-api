import os
import logging
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests

load_dotenv()

# -------------------- الإعدادات --------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "gpt2"  # يمكن استبدالها بأي نموذج من HuggingFace (مثلاً "microsoft/DialoGPT-medium")
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------- الاتصال بالخدمات --------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# -------------------- تطبيق FastAPI --------------------
app = FastAPI(title="Linux Command Assistant")

# السماح بالطلبات من أي مصدر (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

@app.get("/")
def home():
    return {"message": "Linux Command RAG API is running. Use /ask?q=your question"}

@app.get("/ask")
def ask(q: str = Query(..., description="Your question about Linux commands")):
    logging.info(f"Received question: {q}")

    # 1. تحويل السؤال إلى embedding
    q_emb = embedder.encode(q).tolist()

    # 2. البحث في Qdrant عن أكثر 5 نتائج شبهاً
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_emb,
        limit=5
    )

    # 3. استخراج النصوص من النتائج
    contexts = [hit.payload["text"] for hit in search_result]
    # يمكنك أيضاً إرجاع الأوامر مباشرة
    commands = [hit.payload["output"] for hit in search_result]

    # 4. بناء الـ prompt لإرساله إلى النموذج
    if contexts:
        context_str = "\n".join(contexts)
        prompt = f"""Based on the following Linux command examples, answer the user's question.
If the question asks for a command, provide the most appropriate command.

Examples:
{context_str}

Question: {q}
Answer:"""
    else:
        prompt = f"Answer the question about Linux commands: {q}"

    # 5. استدعاء HuggingFace Inference API
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.3,
            "return_full_text": False  # لا نريد تكرار prompt في الرد
        }
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        # HuggingFace قد يعيد قائمة من المخرجات
        if isinstance(result, list) and "generated_text" in result[0]:
            answer = result[0]["generated_text"].strip()
        else:
            answer = str(result)
    else:
        answer = f"Error from HuggingFace API: {response.status_code} - {response.text}"

    return {
        "question": q,
        "answer": answer,
        "similar_commands": commands[:3]  # نعرض أشهر 3 أوامر (اختياري)
    }