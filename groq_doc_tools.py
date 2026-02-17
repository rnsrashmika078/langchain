from langchain.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import time

load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(
        ..., description="The Country and city, eg. Sri Lanka, Colombo"
    )


@tool(args_schema=GetWeather)
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 30°C (demo)"


model_with_tools = model.bind_tools([get_weather])
ai_msg = model_with_tools.invoke("what is the weather for colombo sri lanka?")

if ai_msg.tool_calls[0]["name"] == "get_weather":
    result = get_weather.invoke(ai_msg.tool_calls[0]["args"])
  
    for chunk in result:
        print(chunk, end="", flush=True)
        time.sleep(0.1)
