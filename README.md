# MiniSense — Survey Analysis Agent

## Setup & Run

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Akansha-Mulchandani/MiniSense-.git
cd MiniSense-
python -m venv venv
venv\Scripts\activate          # Windows
# or: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Generate Data & Train Classifier (Part 3 - First Time Only)
```bash
python generate_data.py        # generates 100k stratified survey records (~10MB)
python generate_labels.py      # creates train/val/test splits + confidence scores
python finetune.py             # trains distilbert classifier (F1 = 1.0 on test set, ~15 min)
```

### 3. Test the System
```bash
# Evaluate on 3 sample questions
python main.py --eval

# Start API server at http://localhost:8000
python main.py --serve

# Or ask a single question (requires server running)
python main.py --ask "What are the top complaints this month?"
```

### 4. Test API Endpoints
```powershell
# Test /ask endpoint (orchestrator + all agents)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/ask" `
  -ContentType "application/json" `
  -Body '{"question": "What is our CSAT this month vs last month?"}'

# Test /classify endpoint (fine-tuned model)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/classify" `
  -ContentType "application/json" `
  -Body '{"text": "The wait was too long and staff was rude"}'

# Test /health endpoint
Invoke-RestMethod -Method GET -Uri "http://localhost:8000/health"
```

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

## Part 3 — Fine-Tuning Design

### Problem Statement
omniSense processes 10,000 survey responses per day and needs to classify free-text feedback into 8 categories (Positive/Negative × 4 aspects: Food Quality, Wait Time, Staff, Ambiance). Using GPT-4o at scale ($0.03 per request) costs $300/day = $9,000/month — unsustainable. A fine-tuned local model can classify at <$1/month compute cost.

### Data Strategy (Questions 1 & 4)

**Bootstrap Phase:**
- Start with **500 labeled examples** via GPT-4o (one-time cost: ~$15)
- Use rule-based keyword matching to auto-label **100k survey responses** into 8 classes
- Validate label quality via confidence scoring (flag <0.6 confidence for human review)

**Active Learning Iteration:**
- Train initial model on 2,400–3,200 stratified samples (300–400 per class)
- Run model on unlabeled data; select **low-confidence predictions** for human review
- Iteratively expand labeled set to ~5,000–10,000 samples
- Each iteration: +100 manually-labeled samples improve minority class performance

**Data Distribution:** 
- Stratified sampling ensures all 8 classes equally represented (12.5% each)
- Train/Val/Test split: 70/10/20 with stratification per label
- Back-translation augmentation for minority classes if needed (translate Spanish ↔ English)

### Model & Technique Selection (Question 2)

**Base Model:** `distilbert-base-uncased` (66M params)
- Fast (100ms inference on CPU)
- Excellent on short-text classification
- Fits on single GPU with batch size 64
- Inference easily exported to ONNX for production

**Fine-Tuning Approach:** Full FT on classification head + last 2 layers
- Rationale: Task is classification (not generation), so full parameter tuning is appropriate
- Simpler than LoRA, converges faster on small datasets
- For larger models (7B+), use QLoRA (4-bit quantization + LoRA) to fit in 24GB VRAM

### Training Pipeline (Question 3)

**Framework:** Hugging Face `Trainer` + `datasets`
- Batch size: 64 (maximize GPU utilization)
- Learning rate: 2e-5 (DistilBERT standard)
- Epochs: 1–3 (early stopping on validation F1 ≥ 0.88)
- Mixed precision (fp16): 2x speedup on modern GPUs
- Training time: ~15–30 minutes on single GPU

**Experiment Tracking:** MLflow or Weights & Biases free tier
- Log per-class F1, confusion matrices, training loss
- Version models and hyperparameters for reproducibility

### Evaluation & Readiness (Question 4)

**Go/No-Go Metrics:**
- Macro F1 ≥ 0.88 on held-out test set
- Minimum per-class F1 ≥ 0.80 (no class left behind)
- Inference latency <100ms on CPU

**Validation Strategy:**
- Evaluate on stratified test set (20% of labeled data, ~20k samples)
- Manual review of misclassified samples to identify systematic errors
- A/B test: shadow deploy fine-tuned model vs GPT-4o on live traffic for 48 hours
- If fine-tuned accuracy ≥95% of GPT-4o accuracy, approve for production

### Serving Strategy (Question 5)

**Architecture:**
- Export model to ONNX (`optimum` library) for low-latency inference
- Deploy as `/classify` microservice (FastAPI + `transformers` inference)
- Place behind same API gateway as existing `/ask` route
- Use feature flag (`classifier_v2`) to route traffic without disrupting other routes

**Fallback:**
- If fine-tuned model fails, automatically route to GPT-4o (cost recovery)
- Monitor inference latency and accuracy; auto-revert if <95% accuracy

### Future-Proofing (Question 6)

**Input/Output Agnosticity:**
- Training script reads `column_map.yaml` (not hardcoded field names)
- Supports any `{input_text, label}` schema — swappable data sources
- Inference interface: `Classifier.predict(text: str) → {label, confidence}` (model-agnostic)

**Iterative Improvement:**
- Monthly re-training on accumulated labeled data
- Active learning loop: low-confidence predictions → human review → retrain
- Version control: save all models + training configs for rollback

### Cost Analysis & Production Scaling (Implicit)

**Per-Response Cost Breakdown:**
- Fine-tuned model: $0.00001/request (negligible compute)
- vs. GPT-4o: $0.00003/request
- Daily savings: 10,000 × $0.00002 = $0.20/day = $73/year
- ROI: Positive after 2 months of operation

**Monthly Training Cost:**
- GPU compute (1 epoch, 70k samples): ~$2 (spot instance)
- Human labeling (active learning): ~$50 (for 500 edge cases)
- Total: <$100/month vs. $9,000 for GPT-4o

 

---

## Expected Results After Running

After completing all steps above, you should see:

1. **Data Generation** (`generate_data.py`):
   - Created: `data/survey_responses.json` (100k samples, all 8 classes balanced)
   
2. **Labeling** (`generate_labels.py`):
   - Created: `data/labeled_responses_train.jsonl` (70k samples)
   - Created: `data/labeled_responses_val.jsonl` (10k samples)
   - Created: `data/labeled_responses_test.jsonl` (20k samples)
   - Created: `data/column_map.yaml` (label ID mapping)
   
3. **Training** (`finetune.py`):
   - Trained model saved to: `models/classifier/`
   - Output: `Macro F1 = 1.0000 on test set` (all 8 classes achieved F1 = 1.0)
   - Training time: ~15 minutes on GPU
   
4. **API Running**:
   - POST `/ask` returns orchestrator response with data + RAG + summary
   - POST `/classify` returns `{label, confidence}` from fine-tuned model
   - GET `/health` returns `{status: "ok"}`

---

## What I skipped and why
- **Local LLM for narrative synthesis:** Replaced with a deterministic template engine. A local Ollama/llama3 call could be added in `summary_agent.py` in ~10 lines — but it requires the user to have Ollama running, which breaks "minimal setup" for an assessment. The template produces accurate, readable answers for structured data questions.
- **Streaming / async:** Out of scope for this assessment; FastAPI routes are sync for simplicity.
- **Authentication:** Not relevant for a local assessment server.
