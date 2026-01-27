from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # Get API key from environment
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("Please set GOOGLE_API_KEY in your .env file")
        return
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Use HuggingFace embeddings (FREE, runs locally)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            st.success("PDF processed successfully!")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            google_api_key=google_api_key
        )
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            with st.spinner("Searching for answer..."):
                # Get relevant documents
                docs = knowledge_base.similarity_search(user_question, k=3)
                
                # Combine the context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create a simple prompt
                prompt = f"""Based on the following context from the PDF,return.

Context:
{context}

Question: {user_question}

Answer:"""
                
                # Get response from LLM
                response = llm.invoke(prompt)
                
                # Display the answer
                st.write(response.content)


if __name__ == '__main__':
    main()