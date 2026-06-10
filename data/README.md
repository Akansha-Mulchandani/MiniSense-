# Data Generation & Labeling Strategy

## Overview

This directory contains survey response data and labeled training data for the MiniSense classifier. The generation and labeling process is designed with careful attention to reproducibility, class balance, and data quality.

## Data Generation (`generate_data.py`)

### Strategy: Stratified Class Distribution

**Problem Addressed:**
- Random sampling with rating weights (5%, 10%, 20%, 35%, 30% for ratings 1-5) creates **imbalanced classes**
- Minority classes (Negative feedback) would be underrepresented
- This hurts model training, as some classes have <10% representation

**Solution: Stratified Generation**
- Generate **exactly 12,500 samples per class** (100k ÷ 8 = 12,500)
- Map sentiment → rating range:
  - **Positive** (ratings 4-5) → 25,000 samples
  - **Negative** (ratings 1-2) → 25,000 samples
  - **Neutral** (rating 3) → 50,000 samples
- Each class gets equal representation

### Templates

Aspect-specific free-text templates ensure:
- **Semantic consistency**: Food quality reviews mention food, not staff
- **Sentiment alignment**: Positive reviews use positive language
- **Realism**: Multiple variations prevent memorization

Templates are defined per aspect and sentiment to ensure the generated text naturally aligns with the intended label.

### Reproducibility

- `random.seed(42)` ensures deterministic generation
- Date range: April 1 - May 31, 2026 (2 months)
- Business/survey/channel distribution: uniform random

## Label Generation (`generate_labels.py`)

### Classification Logic

Each survey response is classified into one of **8 categories**:
- 4 aspects: Food Quality, Wait Time, Staff, Ambiance
- 2 sentiments: Positive, Negative
- Labels: `{Sentiment} – {Aspect}`

#### Sentiment Determination

| Rating | Logic |
|--------|-------|
| 4-5 | **Positive** (strong signal) |
| 1-2 | **Negative** (strong signal) |
| 3 | Keyword voting (fallback to Positive if tie) |

#### Aspect Determination

Uses keyword matching with weighted scoring:
- **Regular keywords**: 1 point each (e.g., "wait", "food")
- **Strong keywords**: 2 points each (e.g., "wait time", "food quality")
- Fallback to **"Staff"** if no keywords match

**Known Limitation:** 
- Multi-aspect reviews default to the aspect with highest keyword count
- This may misclassify some mixed feedback (e.g., "food was great but staff was rude" → correctly picks highest match)

### Confidence Scoring (0.5–1.0)

Confidence reflects label reliability for **active learning**:

```
Base confidence: 0.5

Sentiment bonus:
  - Ratings 4-5 or 1-2: +0.3 (strong agreement)
  - Rating 3 with keyword support: +0.1–0.2

Aspect bonus:
  - Found strong keywords: +0.05–0.2
  - No keyword match (fallback): +0.05
```

**Low-confidence samples (<0.6):**
- Candidates for human review
- Typically: ambiguous rating 3 responses or multi-aspect reviews
- ~15–20% of data normally falls in this range
- Marked for active learning iteration

### Train/Val/Test Split (70/10/20)

**Stratified splitting ensures:**
- All 8 classes represented in each split
- Each label proportionally distributed
- No label appears in only one split

**Process:**
1. Group labeled data by label (8 groups)
2. For each group: stratified split into train (70%), temp (30%)
3. Split temp: 50/50 into val (10%) and test (20%)
4. Shuffle each split independently

**Result:**
- Train: ~70,000 samples (12,500 per class × 8 ÷ 1.428)
- Val: ~10,000 samples
- Test: ~20,000 samples

## Files

### Generated Files

```
data/
├── survey_responses.json              # 100k raw survey responses
├── labeled_responses_train.jsonl      # 70k labeled + confidence (train)
├── labeled_responses_val.jsonl        # 10k labeled + confidence (val)
├── labeled_responses_test.jsonl       # 20k labeled + confidence (test)
├── column_map.yaml                    # Label mapping & schema
└── README.md                          # This file
```

### JSONL Format

Each line is a JSON object:

```json
{
  "input_text": "The food was great but the wait time was too long.",
  "label": "Positive – Food Quality",
  "label_id": 0,
  "confidence": 0.85,
  "rating": 4
}
```

### column_map.yaml Format

```yaml
input_field: input_text
label_field: label
label_map:
  "Negative – Ambiance": 7
  "Negative – Food Quality": 1
  "Negative – Staff": 5
  "Negative – Wait Time": 3
  "Positive – Ambiance": 6
  "Positive – Food Quality": 0
  "Positive – Staff": 4
  "Positive – Wait Time": 2
```

## Known Limitations & Future Improvements

### Current Limitations

1. **Simple keyword matching** may not handle sarcasm or complex language
   - Example: "Great wait time!" (sarcasm) might be classified as positive
   - Fix: Could use sentiment analysis model for edge cases

2. **Multi-aspect reviews default to highest keyword count**
   - May lose information for complex feedback
   - Fix: Could generate multi-label data or track aspect hierarchy

3. **Generated templates are synthetic**
   - Real survey data may have more varied language
   - Mitigated by template diversity, but domain shift expected

4. **Fixed rating → sentiment mapping**
   - Doesn't account for context ("Good food but awful service" with rating 3)
   - Rating 3 fallback to keywords reduces this impact

### Future Improvements (Active Learning)

1. **Human review phase** (collect 200–300 high-disagreement samples)
2. **Model-based selection** (run preliminary model, flag low-confidence predictions)
3. **Back-translation augmentation** (for minority classes if needed)
4. **Iterative refinement** (re-label edge cases, update keywords)

## How to Regenerate

### Generate raw survey data (100k responses):
```bash
python generate_data.py
```

Output: `data/survey_responses.json`

### Generate labels with train/val/test split:
```bash
python generate_labels.py
```

Output:
- `data/labeled_responses_train.jsonl`
- `data/labeled_responses_val.jsonl`
- `data/labeled_responses_test.jsonl`
- `data/column_map.yaml`

### Force regeneration (overwrites existing):
```bash
python generate_labels.py --force
```

## Statistics

After running the scripts:

### Survey Generation
- Total: 100,000 responses
- Date range: April–May 2026
- Businesses: 3
- Survey types: 3
- Response channels: 4 (mobile, web, kiosk, email)

### Labeling Statistics
- Classes: 8 (balanced)
- Stratified splits: train (70%), val (10%), test (20%)
- Confidence: 0.5–1.0
- Low-confidence (<0.6): ~15–20%

---

*Last updated: June 2026*
*Reproducible with `random.seed(42)`*
