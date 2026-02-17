from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)
messages = [
    ("system", "You are a helpful geography teacher."),
    ("human", "describe the country SRI LANKA in 3 lines."),
]
stream = model.stream(messages)
full = None

for chunk in stream:
    print(chunk.content, end="", flush=True)
    time.sleep(0.1)
    
    if full is None:
        full = chunk
    else:
        full += chunk

print(f"\n\nTotal Tokens: {full.usage_metadata.get('total_tokens')}")