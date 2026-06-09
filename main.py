"""
FastAPI app for MiniSense. Also runnable as CLI.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import warnings
import json
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator import OrchestratorInput, run as orchestrate

# Suppress warnings and telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['SENTENCE_TRANSFORMERS_NO_TELEMETRY'] = '1'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*telemetry.*')

app = FastAPI(title="MiniSense Survey Analysis Agent")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QuestionRequest):
    result = orchestrate(OrchestratorInput(question=req.question))
    return result.model_dump()

@app.get("/health")
def health():
    return {"status": "ok"}

# --- Sample eval questions ---
SAMPLE_QUESTIONS = [
    "What are the top 3 complaints this month and how do they compare to last month?",
    "What is our CSAT score for May and how does it relate to our target?",
    "What are the main reasons customers are unhappy with wait time?",
]

def run_eval():
    print("\n=== MiniSense Evaluation: 3 Sample Questions ===\n")
    from rag_pipeline import get_collection  # trigger ingest if needed
    get_collection()
    for q in SAMPLE_QUESTIONS:
        print(f"QUESTION: {q}")
        result = orchestrate(OrchestratorInput(question=q))
        print(f"\nSUB-TASKS DISPATCHED:")
        for st in result.sub_tasks:
            print(f"  -> {st['agent']}: {json.dumps(st['task_spec'])}")
        print(f"\nRAG CHUNKS RETRIEVED:")
        for chunk in result.rag_output["retrieved_chunks"]:
            print(f"  [{chunk['chunk_id']}] {chunk['text'][:100]}...")
        print(f"\nFINAL ANSWER:\n{result.final_answer}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation with 3 sample questions")
    parser.add_argument("--ask", type=str, help="Ask a single question")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    args = parser.parse_args()

    if args.eval:
        run_eval()
    elif args.ask:
        from rag_pipeline import get_collection
        get_collection()
        result = orchestrate(OrchestratorInput(question=args.ask))
        print(f"\nFINAL ANSWER:\n{result.final_answer}")
    elif args.serve:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        run_eval()
