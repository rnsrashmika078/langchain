# pip install -qU langchain "langchain[anthropic]"
# pip install -qU langchain langchain-groq ( -q less console output , U -> upgrade )
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

load_dotenv()  # load environment variables

basic_model = ChatGroq(
    model="llama-3.1-8b-instant", temperature=1, timeout=30, max_tokens=50
)
advanced_model = ChatGroq(
    model="openai/gpt-oss-120b", temperature=1, timeout=30, max_tokens=100
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity"""
    message_count = len(request.state["messages"])
    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))


agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection],
    system_prompt="You are a helpful assistant.answer in less word",
)
history = []
while True:
    prompt = input("ask a question?\n")
    history.append({"role": "user", "content": prompt})

    result = agent.invoke({"messages": history})

    ai_msg = result["messages"][-1]
    history.append({"role": "assistant", "content": ai_msg.content})

    model = ai_msg.response_metadata["model_name"]
    total_tokens = ai_msg.response_metadata["token_usage"]["total_tokens"]
    print(f"{ai_msg.content} ({total_tokens}) - {model}")

