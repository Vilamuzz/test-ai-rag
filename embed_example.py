from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

texts = [
    "The Eiffel Tower is located in Paris.",
    "Bananas are a good source of potassium.",
    "Python is a popular programming language."
]

def embed(text):
    response = client.embeddings.create(
        model="mxbai-embed-large",
        input=text
    )
    return response.data[0].embedding

for t in texts:
    vec = embed(t)
    print(f"Text: {t}")
    print(f"Vector (first 5 dims): {vec[:5]}\n")
