import asyncio
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class GetWeather(BaseModel):
    location: str = Field(..., description="Country and city")


@tool(args_schema=GetWeather)
def get_weather(location: str) -> str:
    """(pretend as actual weather) weather in specific location"""
    return f"Weather in {location}: Sunny, 30°C"


async def main():
    prompt = input("\nAsk question: ")
    tools = [get_weather]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
        You are a helpful assistant.
        Only use the tools I have provided. 
        Do NOT invent any new tools. 
        Answer directly if you don’t have a tool for the task.
        """,
    )

    response = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})

    for msg in response["messages"]:
        if msg.type == "tool":
            print("Tool Response:", msg.content)

    print(response["messages"][-1].content)


asyncio.run(main())