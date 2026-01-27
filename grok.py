from dotenv import load_dotenv
from langchain_xai import ChatXAI

load_dotenv()

llm = ChatXAI(
    model="grok-4",
    temperature=0.7,
)

response = llm.invoke("Explain quantum entanglement like I'm 12")
print(response.content)
