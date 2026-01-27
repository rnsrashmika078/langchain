from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, streaming=True)
prompt = input("ask a question\n")
while True:
    response = llm.invoke(prompt)
    totalToken = response.usage_metadata["total_tokens"]
    print(f"Response: {response.content} (token: {totalToken})")
    prompt = input("ask a question\n")
