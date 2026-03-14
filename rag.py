'''
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# 1. Load document
loader = TextLoader("docs/info.txt", encoding="utf-8")
documents = loader.load()

# 2. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

# 3. Embeddings (LOCAL via Ollama)
embeddings = OllamaEmbeddings(model="llama2")

# 4. Vector store (persisted)
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="chroma_db"
)

# 5. Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 6. LLM
llm = Ollama(model="llama2")

print("✅ RAG ready. Type 'exit' to quit.\n")

while True:
    question = input("Ask> ")
    if question.lower() == "exit":
        break

    # Retrieve relevant docs
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an assistant. Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""

    answer = llm.invoke(prompt)
    print("\nAnswer:", answer, "\n")
'''