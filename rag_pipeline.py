"""
RAG pipeline: ingest product FAQ into ChromaDB, retrieve top-k chunks.
Uses sentence-transformers (free, local) for embeddings.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

MODEL_NAME = "all-MiniLM-L6-v2"
FAQ_PATH = "data/product_faq.txt"
COLLECTION_NAME = "product_faq"
CHUNK_SIZE = 200  # words per chunk, sentence-aware

_model = None
_collection = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        _collection = client.get_collection(COLLECTION_NAME)
        if _collection.count() > 0:
            return _collection
    except Exception:
        pass
    _collection = client.get_or_create_collection(COLLECTION_NAME)
    ingest_faq(_collection)
    return _collection

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Sentence-aware chunking: split on Q: boundaries first, then by word count."""
    # Split on Q&A boundaries
    import re
    blocks = re.split(r'\n(?=Q:)', text.strip())
    chunks = []
    for block in blocks:
        words = block.split()
        if len(words) <= chunk_size:
            chunks.append(block.strip())
        else:
            # further split by sentences
            sentences = re.split(r'(?<=[.!?])\s+', block)
            current, count = [], 0
            for s in sentences:
                wc = len(s.split())
                if count + wc > chunk_size and current:
                    chunks.append(" ".join(current))
                    current, count = [s], wc
                else:
                    current.append(s)
                    count += wc
            if current:
                chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]

def ingest_faq(collection):
    model = get_model()
    with open(FAQ_PATH, "r") as f:
        text = f.read()
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Ingested {len(chunks)} chunks into ChromaDB.")

def retrieve(query: str, top_k: int = 3) -> list[dict]:
    collection = get_collection()
    model = get_model()
    query_emb = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text": doc,
            "distance": results["distances"][0][i] if "distances" in results else None,
        })
    return chunks
