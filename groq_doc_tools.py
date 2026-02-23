from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

class GetWeather(BaseModel):
    location: str = Field(..., description="Country and city")

@tool(args_schema=GetWeather)
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 30°C"

tools = [get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_tools_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({
    "input": "What's the weather in Colombo, Sri Lanka?"
})

print(response["output"])