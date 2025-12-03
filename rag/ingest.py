import os
import numpy as np
import faiss
from .embeddings import embed
from .chunking import chunk_text

def ingest_documents(docs: list[str], chunk_size=500, overlap=50):
    all_chunks = []
    chunk_sources = []  # track which document + chunk index
    
    for idx, doc in enumerate(docs):
        chunks = chunk_text(doc, chunk_size, overlap)
        for c_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_sources.append((idx, c_idx))
    
    # Generate embeddings
    vectors = np.array([embed(chunk) for chunk in all_chunks], dtype='float32')
    
    # Build FAISS
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, all_chunks, chunk_sources
