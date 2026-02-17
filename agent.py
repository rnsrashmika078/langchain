# pip install -qU langchain "langchain[anthropic]"
# pip install -qU langchain langchain-groq ( -q less console output , U -> upgrade )
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()  # load environment variables

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=1,timeout=30,max_tokens=50)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=1,timeout=30,max_tokens=50)

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant.answer in less word",
    
    
)
while True:
    prompt = input("ask a question?\n")
    result = agent.invoke({"messages": [{"role": "user", "content": f"{prompt}"}]})
    reply = result["messages"][-1].content
    total_token = result["messages"][1].response_metadata["token_usage"]["total_tokens"]
    print(f"{reply} ({total_token})")

# Run the agent
