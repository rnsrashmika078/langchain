# agent_doc.py
import os
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ FIXED

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load .env file (if present)
load_dotenv()

# 1️⃣ Load PDF
def load_pdf(path):
    try:
        reader = PdfReader(path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        raise Exception(f"❌ Failed to read PDF: {e}")

# 2️⃣ Get API key (Priority: .env > env var > prompt)
def get_groq_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        print("⚠️  GROQ_API_KEY not found in .env or environment.")
        from getpass import getpass
        key = getpass("🔑 Enter your Groq API key: ").strip()
    return key

# === MAIN ===
PDF_PATH = "doc.pdf"
DB_PATH = Path("faiss_index")

# Ensure PDF exists
if not os.path.exists(PDF_PATH):
    print(f"❌ PDF not found: {PDF_PATH}")
    exit(1)

# Get API key
GROQ_API_KEY = get_groq_key()
if not GROQ_API_KEY:
    print("❌ No API key provided. Please set GROQ_API_KEY in .env or enter manually.")
    exit(1)

# Load & chunk PDF
print("\n📄 Loading PDF...")
text = load_pdf(PDF_PATH)

print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(text)
print(f"✅ Created {len(chunks)} chunks.")

# Embed with HuggingFace (all-MiniLM-L6-v2)
print("🧠 Embedding (first time only)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load or create FAISS index
if DB_PATH.exists():
    print("📂 Loading FAISS index from disk...")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    print("💾 Creating new FAISS index...")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(DB_PATH)

# Initialize Groq LLM
print("🚀 Starting Groq LLM (openai/gpt-oss-120b)...")
llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=GROQ_API_KEY)

# Chat loop
print("\n💬 Chatting with your PDF! (Type 'quit' or Ctrl+C to exit)\n")

retriever = db.as_retriever(k=3)

try:
    while True:
        question = input("👤 You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        # Retrieve top 3 chunks
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Simple prompt
        prompt = f"""Use ONLY this context to answer the question. If unsure, say: "I don't know.".

Context:
{context}

Question: {question}
Answer:"""

        # Get response
        response = llm.invoke(prompt)
        print(f"🤖 Assistant: {response.content}\n")

except KeyboardInterrupt:
    print("\n👋 Goodbye!")
except Exception as e:
    print(f"\n⚠️ Error: {e}")
