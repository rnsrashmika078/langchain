from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

def get_weather_data(location: str):
    return {"location": location, "temp": 30, "humidity": 80}

weather_runnable = RunnableLambda(get_weather_data)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Weather report for {location}.Give advice in 2 lines.")
])

chain = weather_runnable | prompt | model

res = chain.invoke("Colombo")
print(res.content)
