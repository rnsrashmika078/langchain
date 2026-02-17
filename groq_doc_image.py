from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import time

load_dotenv()
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
messages = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image in 10 word"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://res.cloudinary.com/dwcjokd3s/image/upload/v1758806505/Listing/gbo3afudm0kmmpr4olzd.jpg"
            },
        },
    ]
)

stream = model.stream([messages])

for chunk in stream:
    print(chunk.content, end="", flush=True)
    time.sleep(0.1)
# print(response.content)
