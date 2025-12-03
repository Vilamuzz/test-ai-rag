from .documents import documents
from .ingest import ingest_documents
from .embeddings import embed

print("Building index with chunking...")
index, chunks, chunk_sources = ingest_documents(documents)
print(f"Total chunks: {len(chunks)}")

def search(query: str, k: int = 3):
    q_vec = embed(query).reshape(1, -1)
    distances, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0]]
