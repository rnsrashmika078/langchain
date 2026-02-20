from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, Chroma



loader = PyPDFLoader("./doc.pdf")
document = loader.load()

print(f"Content of doc {document[0].page_content}")
