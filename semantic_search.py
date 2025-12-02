import faiss
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

documents = [
    "FastAPI is a modern Python web framework.",
    "Ollama allows running LLMs on your local machine.",
    "Embeddings convert text into numerical vectors.",
    "Python is widely used for machine learning."
]

def embed(text):
    response = client.embeddings.create(
        model="mxbai-embed-large",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# Build index
dim = len(embed("hello"))
index = faiss.IndexFlatL2(dim)

doc_vectors = np.array([embed(doc) for doc in documents])
index.add(doc_vectors)

print("Index size:", index.ntotal)

# Query
query = "What is FastAPI used for?"
q_vec = embed(query).reshape(1, -1)

distances, ids = index.search(q_vec, k=2)

print("\nTop results:\n")
for i in ids[0]:
    print("-", documents[i])
