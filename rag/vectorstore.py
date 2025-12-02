import faiss
import numpy as np
from .embeddings import embed
from .documents import documents

# Build FAISS index once on startup
def build_index():
    print("Building FAISS index...")

    sample_vec = embed("hello")
    dim = len(sample_vec)

    index = faiss.IndexFlatL2(dim)

    vectors = np.array([embed(doc) for doc in documents], dtype="float32")
    index.add(vectors)

    print("Index built. Total docs:", index.ntotal)
    return index, vectors

index, doc_vectors = build_index()

# Search function
def search(query: str, k: int = 3):
    q_vec = embed(query).reshape(1, -1)
    distances, ids = index.search(q_vec, k)
    return ids[0]   # list of doc indexes
