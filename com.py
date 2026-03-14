'''
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
import json

def load_linux_commands(json_path):
    documents = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        content = f"""
User request:
{item['input']}

Linux command:
{item['output']}
"""
    documents.append(
        Document(
            page_content=content,
            metadata={"type": "linux_command"}
        )
    )

    return documents


documents = load_linux_commands("./archive/complex_linux_commands_million.json")


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="llama2")

db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="linux_chroma_db"
)


# 3. Embeddings (LOCAL via Ollama)
embeddings = OllamaEmbeddings(model="llama2")


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
You are a Linux expert.
Convert the user request into a correct Linux command.
Answer ONLY with the command.

Examples:
{context}

User request:
{question}
"""


    answer = llm.invoke(prompt)
    print("\nAnswer:", answer, "\n")

    

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json


# -----------------------
# LOAD JSON
# -----------------------

def load_linux_commands(json_path):

    documents = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # مهم جداً لا تأخذ مليون
    for item in data[:500]:

        content = f"""
User request:
{item['input']}

Linux command:
{item['output']}
"""

        documents.append(
            Document(
                page_content=content,
                metadata={"type": "linux_command"}
            )
        )

    return documents


documents = load_linux_commands(
    "./archive/complex_linux_commands_million.json"
)


# -----------------------
# SPLITTER (مهم)
# -----------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)


# -----------------------
# EMBEDDINGS (بدل llama2)
# -----------------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# -----------------------
# CHROMA
# -----------------------

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="linux_chroma_db"
)

db.persist()


# -----------------------
# RETRIEVER
# -----------------------

retriever = db.as_retriever(
    search_kwargs={"k": 5}
)


# -----------------------
# LLM
# -----------------------

llm = Ollama(model="llama2")


print("RAG ready. type exit\n")


# -----------------------
# LOOP
# -----------------------

while True:

    question = input("Ask> ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=5)

    print("\nDEBUG:\n")

    for d in docs:
        print(d.page_content[:80])
        print("------")

    context = "\n\n".join(
        d.page_content for d in docs
    )

    prompt = f"""
You are a Linux expert.

Use examples to generate command.

Examples:
{context}

User request:
{question}

Command:
"""

    answer = llm.invoke(prompt)

    print("\nAnswer:", answer, "\n")    
'''
