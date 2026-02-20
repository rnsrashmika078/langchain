# agent_doc_simple.py
from pathlib import Path
from io import BytesIO
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import Chroma
import time

load_dotenv()
# url = "https://res.cloudinary.com/dwcjokd3s/image/upload/v1771502897/zg9jmnvxjpmlbu17on7c.pdf"
url = "https://res.cloudinary.com/dwcjokd3s/image/upload/v1771607902/LiveLink/uploads/snvtpgt5od7zyaeqydmi.pdf"

DB_PATH = Path("chroma_db_" + url.split("/")[-1])
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Load PDF
def load_pdf(path):
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)


# Split into chunks

response = requests.get(url)
pdf_file = BytesIO(response.content)
text = load_pdf(pdf_file)
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(
    text
)

# Create or load CHROMA vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if DB_PATH.exists():
    db = Chroma(persist_directory=str(DB_PATH), embedding_function=embeddings)
else:
    db = Chroma.from_texts(chunks, embeddings, persist_directory=str(DB_PATH))
    db.persist()  # save the database


# Initialize Groq LLM
# llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
retriever = db.as_retriever(k=3)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)
question = input("You:").strip()
docs = retriever.invoke(question)
context = "\n\n".join(doc.page_content for doc in docs)
prompt = f"related content from the pdf: {context}. answer the question in short and few words"
stream = llm.stream(prompt)

for chunk in stream:
    print(chunk.content, end="", flush=True)
    time.sleep(0.1)
# Chat loop
while True:
    question = input("You ( MSG ): ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Use ONLY this context to answer the question. If unsure, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    print("Assistant:", llm.invoke(prompt).content)
