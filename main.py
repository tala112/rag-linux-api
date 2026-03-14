import os
import logging
from fastapi import FastAPI, Query, HTTPException
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
HF_MODEL = "gpt2"  # يمكن تغييره لاحقاً
COLLECTION_NAME = "linux_commands"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------- التحقق من المتغيرات --------------------
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("⚠️ تأكدي من ضبط QDRANT_URL و QDRANT_API_KEY في ملف .env")
if not HF_API_TOKEN:
    logging.warning("⚠️ HF_API_TOKEN غير موجود في .env - قد تواجهين مشاكل في الطلبات")

# -------------------- الاتصال بالخدمات --------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL)

# -------------------- تطبيق FastAPI --------------------
app = FastAPI(title="Linux Command Assistant")

# CORS
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

    try:
        # 1. تحويل السؤال إلى embedding
        q_emb = embedder.encode(q).tolist()

        # 2. البحث في Qdrant
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_emb,
            limit=5
        )
        # التأكد من وجود نقاط
        if not response or not response.points:
            logging.warning("No results found in Qdrant")
            contexts = []
            commands = []
        else:
            search_result = response.points
            contexts = [hit.payload.get("text", "") for hit in search_result if hit.payload]
            commands = [hit.payload.get("output", "") for hit in search_result if hit.payload]

        # 3. بناء الـ prompt
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

        # 4. استدعاء HuggingFace API
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.3,
                "return_full_text": False
            }
        }

        '''hf_response = requests.post(
         f"https://router.huggingface.co/models/{HF_MODEL}",
         headers=headers,
         json=payload,
         timeout=30
        )'''
        
        hf_response = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )

        if hf_response.status_code == 200:
            result = hf_response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                answer = result[0]["generated_text"].strip()
            else:
                answer = str(result)
        else:
            answer = f"Error from HuggingFace API: {hf_response.status_code} - {hf_response.text}"

        return {
            "question": q,
            "answer": answer,
            "similar_commands": commands[:3]
        }

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
'''
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
    search_result = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=q_emb,          # هنا التغيير المهم
    limit=5
)
    search_result = response.points
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
'''