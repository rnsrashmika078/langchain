# pip install -qU langchain "langchain[anthropic]"
# pip install -qU langchain langchain-groq ( -q less console output , U -> upgrade )
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import tool

load_dotenv()  # load environment variables

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=1,timeout=30,max_tokens=20)

agent = create_agent(
    model=llm,
    tools=[get_weather, search],
    system_prompt="You are a helpful assistant. always reply in less words. use given tools as possible",
)
while True:
    prompt = input("ask a question?\n")
    result = agent.invoke({"messages": [{"role": "user", "content": f"{prompt}"}]})
    reply = result["messages"][-1].content
    total_token = result["messages"][1].response_metadata["token_usage"]["total_tokens"]
    print(f"{reply} ({total_token})")

# Run the agent
