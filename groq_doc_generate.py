from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)
messages = [
    ("system", "You are a helpful math teacher."),
    ("human", "1 + 1."),
]
response = model.invoke(messages)
print(f"{response.content} : ({response.response_metadata['token_usage']['total_tokens']})")
