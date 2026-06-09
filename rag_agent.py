"""
RAGAgent: retrieves relevant FAQ chunks for a given query.
Returns structured output — no free-form text.
"""
from pydantic import BaseModel
from rag_pipeline import retrieve

class RAGAgentInput(BaseModel):
    query: str
    top_k: int = 3

class RAGAgentOutput(BaseModel):
    query: str
    retrieved_chunks: list[dict]  # [{chunk_id, text, distance}]

def run(inp: RAGAgentInput) -> RAGAgentOutput:
    chunks = retrieve(inp.query, top_k=inp.top_k)
    return RAGAgentOutput(query=inp.query, retrieved_chunks=chunks)
