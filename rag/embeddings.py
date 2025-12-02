import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="mxbai-embed-large",
        input=text
    )
    vec = response.data[0].embedding
    return np.array(vec, dtype="float32")
