"""
DataAgent: parses survey JSON, computes exact metrics.
Returns structured JSON — no free-form text.
"""
import json
from collections import Counter, defaultdict
from pydantic import BaseModel
from typing import Optional
import re

_responses = None

def load_responses():
    global _responses
    if _responses is None:
        with open("data/survey_responses.json") as f:
            _responses = json.load(f)["responses"]
    return _responses

# --- Tool functions (called by DataAgent) ---

def compute_csat(responses: list) -> float:
    """CSAT% = satisfied (rating >= 4) / total * 100"""
    if not responses:
        return 0.0
    satisfied = sum(1 for r in responses if r["rating"] >= 4)
    return round(satisfied / len(responses) * 100, 2)

def compute_avg_rating(responses: list) -> float:
    if not responses:
        return 0.0
    return round(sum(r["rating"] for r in responses) / len(responses), 3)

def extract_themes(responses: list, top_n: int = 5) -> list[dict]:
    """Extract top themes from free_text using keyword matching."""
    THEME_KEYWORDS = {
        "wait time": ["wait", "slow", "queue", "long", "delay"],
        "food quality": ["food", "meal", "taste", "delicious", "bland", "fresh", "stale"],
        "staff": ["staff", "service", "rude", "friendly", "helpful", "attitude"],
        "cleanliness": ["clean", "dirty", "hygiene", "tidy"],
        "pricing": ["price", "expensive", "cheap", "value", "cost", "worth"],
        "ambiance": ["atmosphere", "ambiance", "music", "noise", "loud", "quiet"],
    }
    counts = Counter()
    for r in responses:
        text = r["free_text"].lower()
        for theme, kws in THEME_KEYWORDS.items():
            if any(kw in text for kw in kws):
                counts[theme] += 1
    total = len(responses) if responses else 1
    return [
        {"theme": theme, "count": count, "pct": round(count / total * 100, 1)}
        for theme, count in counts.most_common(top_n)
    ]

def filter_by_period(responses: list, year: int, month: int) -> list:
    return [
        r for r in responses
        if r["date"].startswith(f"{year}-{month:02d}")
    ]

# --- DataAgent class ---

class DataAgentInput(BaseModel):
    task: str  # "metrics", "themes", "comparison", "count"
    period_year: Optional[int] = 2026
    period_month: Optional[int] = None  # None = all data
    compare_month: Optional[int] = None
    top_n: int = 5

class DataAgentOutput(BaseModel):
    task: str
    period_label: str
    total_responses: int
    avg_rating: float
    csat_pct: float
    top_themes: list[dict]
    comparison: Optional[dict] = None

def run(inp: DataAgentInput) -> DataAgentOutput:
    responses = load_responses()

    if inp.period_month:
        subset = filter_by_period(responses, inp.period_year, inp.period_month)
        label = f"{inp.period_year}-{inp.period_month:02d}"
    else:
        subset = responses
        label = "all-time"

    result = DataAgentOutput(
        task=inp.task,
        period_label=label,
        total_responses=len(subset),
        avg_rating=compute_avg_rating(subset),
        csat_pct=compute_csat(subset),
        top_themes=extract_themes(subset, inp.top_n),
    )

    if inp.compare_month:
        compare_subset = filter_by_period(responses, inp.period_year, inp.compare_month)
        result.comparison = {
            "period_label": f"{inp.period_year}-{inp.compare_month:02d}",
            "total_responses": len(compare_subset),
            "avg_rating": compute_avg_rating(compare_subset),
            "csat_pct": compute_csat(compare_subset),
            "top_themes": extract_themes(compare_subset, inp.top_n),
            "rating_delta": round(
                compute_avg_rating(subset) - compute_avg_rating(compare_subset), 3
            ),
            "csat_delta": round(
                compute_csat(subset) - compute_csat(compare_subset), 2
            ),
        }

    return result
