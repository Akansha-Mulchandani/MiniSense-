# MiniSense — Survey Analysis Agent

## Setup & Run

```bash
git clone https://github.com/Akansha-Mulchandani/MiniSense-.git
cd MiniSense-
python -m venv venv
venv\Scripts\activate          # Windows
# or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python generate_data.py        # generates 100k survey records
python main.py --eval          # runs 3 sample eval questions
python main.py --serve         # starts FastAPI at http://localhost:8000
# or ask a single question:
python main.py --ask "What are the top complaints this month?"
```

API usage:
```powershell
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/ask" -ContentType "application/json" -Body '{"question": "What is our CSAT this month vs last month?"}'
```
Response format: `{"question": "...", "sub_tasks": [...], "data_output": {...}, "rag_output": {...}, "final_answer": "..."}`

---

## Architecture

```
User Question
     │
     ▼
Orchestrator Agent (orchestrator.py)
  ├── Parses intent (month, comparison, top-N) — rule-based, no LLM cost
  ├── Builds structured TaskSpec for each sub-agent
  │
  ├──► DataAgent (data_agent.py)
  │       Tools: compute_csat(), compute_avg_rating(), extract_themes(), filter_by_period()
  │       Input:  DataAgentInput (pydantic)
  │       Output: DataAgentOutput (pydantic) — structured JSON
  │       Tool-calling example:
  │       ```python
  │       from data_agent import compute_csat, extract_themes
  ￼       responses = load_responses()
  ￼       csat = compute_csat(responses)
  ￼       themes = extract_themes(responses, top_n=5)
  ￼       ```
  │
  ├──► RAGAgent (rag_agent.py)
  │       Retrieves top-k FAQ chunks from ChromaDB
  │       Embeddings: sentence-transformers/all-MiniLM-L6-v2 (free, local)
  │       Output: RAGAgentOutput (pydantic) — structured chunk list
  │
  └──► SummaryAgent (summary_agent.py)
          Synthesizes DataAgent + RAG output into narrative answer
          Output: SummaryAgentOutput (pydantic)
```

**Key design decisions:**
- All inter-agent communication uses typed Pydantic models (no raw string passing)
- DataAgent exposes explicit tool functions (`compute_csat`, `extract_themes`) that the orchestrator could call directly — demonstrating the tool-calling pattern
- No paid API required — embeddings via sentence-transformers, vector store via ChromaDB (local), LLM synthesis via template engine (upgradeable to local Ollama)
- Intent parsing is rule-based to avoid LLM dependency in the orchestrator loop

---

## RAG Pipeline

**Chunking strategy:** Q&A boundary-aware splitting. The FAQ is structured as Q&A pairs, so we first split on `Q:` boundaries (preserving semantic units), then fall back to word-count chunking (200 words) for any oversized blocks. This outperforms fixed-size chunking because it avoids splitting a question from its answer.

**Encoding fix:** The FAQ file is opened with UTF-8 encoding in `rag_pipeline.py` to handle special characters correctly.

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers) — 384-dim, runs locally, strong performance on short passages.

**Vector store:** ChromaDB with persistent storage (`chroma_db/`). Top-3 retrieval by cosine distance.

**Evaluation of 3 sample questions:**

| Question | Retrieved chunks quality | Notes |
|---|---|---|
| Top 3 complaints vs last month | Retrieved CSAT target + complaint handling chunks | Works well — keyword overlap is high |
| CSAT score vs target |  Retrieved CSAT methodology + target chunk | Excellent — direct match |
| Why unhappy with wait time | Retrieved wait time + peak hours chunk | Good retrieval, but FAQ is thin on root-cause detail |

**Where retrieval falls short:** The FAQ doesn't contain enough operational detail for root-cause questions (e.g., "why is wait time long on Tuesdays?"). Adding more operational data to the FAQ corpus would improve this.

---

## Fine-Tuning Design (Part 3)

### Task
Classify 10,000 survey responses/day into 8 categories (e.g., *Positive – Food Quality*, *Negative – Wait Time*) without GPT-4o costs.

### Data Strategy
- Bootstrap ~500 labeled examples using GPT-4o (one-time cost), stratified across all 8 classes
- Use active learning: run the fine-tuned model on unlabeled data, select low-confidence predictions for human review — efficiently grow to ~2,000–3,000 labeled examples
- Augment with back-translation (translate to Spanish → back to English) to increase diversity on minority classes
- **Estimate needed:** ~300–400 examples per class = ~2,400–3,200 total for solid performance on a 8-class classification task with short texts

### Model & Technique
- **Base model:** `distilbert-base-uncased` or `roberta-base` — both are fast, small, and excel at short-text classification
- **Technique:** Full fine-tuning of the classification head + last 2 transformer layers (not full FT, not LoRA). Rationale: the task is classification (not generation), so we don't need parameter-efficient generation tuning. Full FT on just the top layers converges fast and is cheap on a single GPU for a ~66M param model
- For larger base models (e.g., Mistral-7B), use QLoRA (4-bit quantization + LoRA) to fit on a single 24GB GPU

### Training Pipeline
- **Tooling:** Hugging Face `Trainer` + `datasets` library
- Training config: batch size 32, lr 2e-5, 5 epochs, early stopping on validation F1
- Data split: 80/10/10 train/val/test, stratified by class
- Experiment tracking: MLflow or W&B (free tier)

### Evaluation
- **Metrics:** Per-class F1, macro F1, confusion matrix
- **Go/no-go threshold:** Macro F1 ≥ 0.88 on held-out test set AND ≥ 0.85 on a manually curated "hard cases" set
- Shadow deploy: run fine-tuned model in parallel with GPT-4o for 48 hours, compare outputs on live traffic before cutover

### Serving
- Export fine-tuned adapter as ONNX for low-latency inference
- Deploy as a separate `/classify` microservice (FastAPI + `optimum` inference) behind the same API gateway
- Use a feature flag to route traffic: `classifier_v2` flag routes to fine-tuned model, keeping existing routes untouched

### Future-Proofing
- Store training data as JSONL with a schema-versioned `input` field (not hardcoded survey field names)
- The training script reads a `column_map.yaml` that maps raw fields to `{input_text, label}` — swapping data sources only requires updating the YAML
- Use a pipeline wrapper class so the inference interface is model-agnostic: `Classifier.predict(text: str) -> {label, confidence}` 

---

## Fine-Tuning Implementation

# Generate labels from existing survey data (rule-based, no API needed)
python generate_labels.py        # creates data/labeled_responses.jsonl

# Fine-tune distilbert on 2400 labeled samples
python finetune.py               # saves model to models/classifier/
                                 # prints macro F1 + per-class F1 on test set

# Classify via API (model must be trained first)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/classify" -ContentType "application/json" -Body '{"text": "The wait was too long and staff was rude"}'
# Returns: {"label": "Negative – Wait Time", "confidence": 0.91}

---

## What I skipped and why
- **Local LLM for narrative synthesis:** Replaced with a deterministic template engine. A local Ollama/llama3 call could be added in `summary_agent.py` in ~10 lines — but it requires the user to have Ollama running, which breaks "minimal setup" for an assessment. The template produces accurate, readable answers for structured data questions.
- **Streaming / async:** Out of scope for this assessment; FastAPI routes are sync for simplicity.
- **Authentication:** Not relevant for a local assessment server.
