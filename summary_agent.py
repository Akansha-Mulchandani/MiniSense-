"""
SummaryAgent: synthesizes DataAgent output + RAG chunks into a narrative answer.
Uses a local LLM via transformers (free). Falls back to template if model unavailable.
"""
from pydantic import BaseModel
from data_agent import DataAgentOutput
from rag_agent import RAGAgentOutput
from typing import Optional
import json

class SummaryAgentInput(BaseModel):
    question: str
    data: DataAgentOutput
    rag: RAGAgentOutput

class SummaryAgentOutput(BaseModel):
    question: str
    narrative: str
    sources_used: list[str]

def _template_answer(inp: SummaryAgentInput) -> str:
    d = inp.data
    compare = d.comparison

    themes_str = ", ".join(
        f"{t['theme']} ({t['pct']}%)" for t in d.top_themes[:3]
    )

    answer = (
        f"For the period {d.period_label}, we received {d.total_responses:,} survey responses. "
        f"The average rating was {d.avg_rating}/5 with a CSAT of {d.csat_pct}%. "
        f"The top feedback themes were: {themes_str}. "
    )

    if compare:
        direction = "improved" if compare["rating_delta"] > 0 else "declined"
        answer += (
            f"Compared to {compare['period_label']}, the average rating {direction} by "
            f"{abs(compare['rating_delta'])} points (CSAT delta: {compare['csat_delta']:+}%). "
        )
        prev_themes = ", ".join(t["theme"] for t in compare["top_themes"][:3])
        answer += f"In the prior period, top themes were: {prev_themes}. "

    if inp.rag.retrieved_chunks:
        answer += (
            "According to the product FAQ, "
            + inp.rag.retrieved_chunks[0]["text"][:200].replace("\n", " ")
            + "..."
        )

    return answer

def run(inp: SummaryAgentInput) -> SummaryAgentOutput:
    narrative = _template_answer(inp)
    sources = [c["chunk_id"] for c in inp.rag.retrieved_chunks]
    return SummaryAgentOutput(
        question=inp.question,
        narrative=narrative,
        sources_used=sources,
    )
