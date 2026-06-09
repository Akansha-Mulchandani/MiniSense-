"""
Orchestrator Agent: receives a natural language business question,
decomposes it into sub-tasks, routes to sub-agents, synthesizes final answer.
"""
import re
from pydantic import BaseModel
from typing import Optional
import data_agent, rag_agent, summary_agent
from data_agent import DataAgentInput
from rag_agent import RAGAgentInput
from summary_agent import SummaryAgentInput

class OrchestratorInput(BaseModel):
    question: str

class OrchestratorOutput(BaseModel):
    question: str
    sub_tasks: list[dict]
    data_output: dict
    rag_output: dict
    final_answer: str

# --- Intent parsing (no LLM needed — rule-based) ---

def _parse_intent(question: str) -> dict:
    q = question.lower()
    intent = {
        "needs_comparison": any(w in q for w in ["compare", "vs", "versus", "last month", "previous", "change", "trend"]),
        "period_month": None,
        "compare_month": None,
        "top_n": 3,
    }

    # Detect month references
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "this month": 5, "current month": 5,
        "last month": 4, "previous month": 4,
    }
    months_found = []
    for phrase, num in month_map.items():
        if phrase in q:
            months_found.append(num)

    if len(months_found) >= 2:
        intent["period_month"] = months_found[0]
        intent["compare_month"] = months_found[1]
    elif len(months_found) == 1:
        intent["period_month"] = months_found[0]
        if intent["needs_comparison"]:
            intent["compare_month"] = months_found[0] - 1 if months_found[0] > 1 else None

    # If no month detected but comparison needed, default to May vs April
    if intent["needs_comparison"] and not intent["period_month"]:
        intent["period_month"] = 5
        intent["compare_month"] = 4

    # top N detection
    match = re.search(r"top\s+(\d+)", q)
    if match:
        intent["top_n"] = int(match.group(1))

    return intent

def run(inp: OrchestratorInput) -> OrchestratorOutput:
    question = inp.question
    intent = _parse_intent(question)

    # --- Build structured sub-task specs ---
    sub_tasks = [
        {
            "agent": "DataAgent",
            "task_spec": {
                "task": "metrics+themes",
                "period_month": intent["period_month"],
                "compare_month": intent["compare_month"],
                "top_n": intent["top_n"],
            },
        },
        {
            "agent": "RAGAgent",
            "task_spec": {
                "query": question,
                "top_k": 3,
            },
        },
    ]

    # --- Execute DataAgent ---
    data_inp = DataAgentInput(
        task="metrics",
        period_year=2026,
        period_month=intent["period_month"],
        compare_month=intent["compare_month"],
        top_n=intent["top_n"],
    )
    data_out = data_agent.run(data_inp)

    # --- Execute RAGAgent ---
    rag_inp = RAGAgentInput(query=question, top_k=3)
    rag_out = rag_agent.run(rag_inp)

    # --- Execute SummaryAgent ---
    summary_inp = SummaryAgentInput(
        question=question,
        data=data_out,
        rag=rag_out,
    )
    summary_out = summary_agent.run(summary_inp)

    return OrchestratorOutput(
        question=question,
        sub_tasks=sub_tasks,
        data_output=data_out.model_dump(),
        rag_output=rag_out.model_dump(),
        final_answer=summary_out.narrative,
    )
