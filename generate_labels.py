"""
Generate labels for survey responses with confidence scoring and stratified train/val/test split.
No API calls needed - fully local.

Data Strategy:
- Classify each response into 8 categories (Positive/Negative × 4 aspects)
- Add confidence scores based on keyword matches
- Create stratified train/val/test split (70%/10%/20%)
- Ensure all 8 classes represented in each split
- Output: 3 separate JSONL files + column_map.yaml
"""
import json
import yaml
import argparse
import random
from collections import Counter, defaultdict

random.seed(42)


def stratified_split(items, test_size=0.3, random_state=42):
    """Simple stratified split without sklearn dependency."""
    random.Random(random_state).shuffle(items)
    split_point = int(len(items) * (1 - test_size))
    return items[:split_point], items[split_point:]

# Label mapping
LABEL_MAP = {
    "Positive – Food Quality": 0,
    "Negative – Food Quality": 1,
    "Positive – Wait Time": 2,
    "Negative – Wait Time": 3,
    "Positive – Staff": 4,
    "Negative – Staff": 5,
    "Positive – Ambiance": 6,
    "Negative – Ambiance": 7,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Keyword mapping with confidence weights
KEYWORD_MAP = {
    "Food Quality": {
        "keywords": ["food", "menu", "taste", "quality", "meal", "delicious", "bland", "fresh", "stale"],
        "strong_keywords": ["food quality", "taste", "delicious", "bland"],
    },
    "Wait Time": {
        "keywords": ["wait", "slow", "time", "queue", "long", "delay", "fast", "quick"],
        "strong_keywords": ["wait time", "long wait", "quick service", "long queue"],
    },
    "Staff": {
        "keywords": ["staff", "service", "rude", "friendly", "helpful", "attitude", "polite"],
        "strong_keywords": ["staff", "service", "rude", "friendly"],
    },
    "Ambiance": {
        "keywords": ["ambiance", "clean", "noise", "atmosphere", "music", "loud", "quiet", "dirty", "tidy"],
        "strong_keywords": ["ambiance", "atmosphere", "music", "clean", "dirty"],
    },
}

POSITIVE_KEYWORDS = ["great", "good", "excellent", "love", "best", "amazing", "fantastic", "perfect", "outstanding", "highly recommend"]
NEGATIVE_KEYWORDS = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing", "unacceptable", "not worth"]


def classify_response_with_confidence(rating: int, free_text: str) -> tuple[str, int, float]:
    """
    Classify response into one of 8 categories with confidence score.
    Returns (label, label_id, confidence_score)
    
    Confidence is based on:
    - Rating agreement with sentiment (high rating = positive sentiment)
    - Keyword matches for aspect
    - Presence of strong keywords
    """
    text_lower = free_text.lower()
    confidence = 0.5  # Base confidence
    
    # === SENTIMENT DETERMINATION ===
    if rating >= 4:
        sentiment = "Positive"
        confidence += 0.3  # Strong rating signal
    elif rating <= 2:
        sentiment = "Negative"
        confidence += 0.3  # Strong rating signal
    else:
        # Rating 3: use keyword fallback to determine sentiment
        pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
            confidence += 0.2 * (pos_count / (pos_count + neg_count + 1))
        elif neg_count > pos_count:
            sentiment = "Negative"
            confidence += 0.2 * (neg_count / (pos_count + neg_count + 1))
        else:
            # Default to Positive for neutral
            sentiment = "Positive"
            confidence += 0.1
    
    # === ASPECT DETERMINATION ===
    aspect_scores = {}
    aspect_confidence_bonus = {}
    
    for aspect, kw_info in KEYWORD_MAP.items():
        # Count regular keywords
        regular_count = sum(1 for kw in kw_info["keywords"] if kw in text_lower)
        # Count strong keywords (more weight)
        strong_count = sum(1 for kw in kw_info["strong_keywords"] if kw in text_lower)
        
        # Score: strong keywords count as 2, regular as 1
        score = strong_count * 2 + regular_count
        aspect_scores[aspect] = score
        
        if score > 0:
            # Higher confidence if we found strong keywords
            aspect_confidence_bonus[aspect] = min(0.2, strong_count * 0.1)
        else:
            aspect_confidence_bonus[aspect] = 0
    
    # Select aspect with highest score, fallback to Staff
    if aspect_scores and max(aspect_scores.values()) > 0:
        aspect = max(aspect_scores, key=aspect_scores.get)
        confidence += aspect_confidence_bonus[aspect]
    else:
        aspect = "Staff"  # Default fallback
        confidence += 0.05  # Lower confidence for defaults
    
    # Clamp confidence to [0.5, 1.0]
    confidence = min(1.0, confidence)
    confidence = max(0.5, confidence)
    
    label = f"{sentiment} – {aspect}"
    label_id = LABEL_MAP[label]
    
    return label, label_id, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regeneration even if labeled data exists")
    args = parser.parse_args()
    
    # Check if labeled data already exists
    train_path = "data/labeled_responses_train.jsonl"
    if not args.force:
        try:
            with open(train_path, "r") as f:
                if f.read().strip():
                    print(f"Labeled data already exists. Use --force to regenerate.")
                    return
        except FileNotFoundError:
            pass
    
    # Load survey responses
    print("Loading survey responses...")
    with open("data/survey_responses.json", "r") as f:
        data = json.load(f)
        responses = data["responses"]
    
    print(f"Processing {len(responses)} responses...")
    
    # Classify each response with confidence
    labeled_data = []
    label_counts = Counter()
    
    for r in responses:
        label, label_id, confidence = classify_response_with_confidence(r["rating"], r["free_text"])
        labeled_data.append({
            "input_text": r["free_text"],
            "label": label,
            "label_id": label_id,
            "confidence": round(confidence, 3),
            "rating": r["rating"],
        })
        label_counts[label] += 1
    
    # === STRATIFIED TRAIN/VAL/TEST SPLIT ===
    print("\nPerforming stratified train/val/test split (70/10/20)...")
    
    # Group by label for stratification
    label_groups = defaultdict(list)
    for item in labeled_data:
        label_groups[item["label"]].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    # For each label, split stratified
    for label, items in label_groups.items():
        # First split: 70% train, 30% temp (will split into val/test)
        train_items, temp_items = stratified_split(
            items, test_size=0.3, random_state=42
        )
        # Second split of temp: 50/50 into val/test (10%/20% of total)
        val_items, test_items = stratified_split(
            temp_items, test_size=0.666, random_state=42  # 2/3 to test = 20% of total
        )
        
        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)
    
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    print(f"Train: {len(train_data)} samples ({100*len(train_data)/len(labeled_data):.1f}%)")
    print(f"Val:   {len(val_data)} samples ({100*len(val_data)/len(labeled_data):.1f}%)")
    print(f"Test:  {len(test_data)} samples ({100*len(test_data)/len(labeled_data):.1f}%)")
    
    # === SAVE SPLITS ===
    print("\nSaving train/val/test splits...")
    
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
    
    split_label_counts = {}
    
    for split_name, split_data in splits.items():
        path = f"data/labeled_responses_{split_name}.jsonl"
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        
        split_counts = Counter(item["label"] for item in split_data)
        split_label_counts[split_name] = split_counts
        print(f"✓ Saved {len(split_data)} samples to {path}")
    
    # === PRINT DISTRIBUTION PER SPLIT ===
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION PER SPLIT")
    print("="*70)
    
    for split_name in ["train", "val", "test"]:
        print(f"\n{split_name.upper()} ({len(splits[split_name])} samples):")
        for label, count in split_label_counts[split_name].most_common():
            pct = 100 * count / len(splits[split_name])
            print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")
    
    # === SAVE COLUMN MAP ===
    column_map_path = "data/column_map.yaml"
    print(f"\nSaving column map to {column_map_path}...")
    with open(column_map_path, "w") as f:
        f.write("input_field: input_text\n")
        f.write("label_field: label\n")
        f.write("label_map:\n")
        for label, label_id in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
            f.write(f'  "{label}": {label_id}\n')
    
    # === PRINT SUMMARY ===
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total labeled responses: {len(labeled_data)}")
    print(f"\nOverall label distribution:")
    for label, count in label_counts.most_common():
        pct = 100 * count / len(labeled_data)
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")
    
    # Confidence statistics
    confidences = [item["confidence"] for item in labeled_data]
    print(f"\nConfidence score statistics:")
    print(f"  Min:  {min(confidences):.3f}")
    print(f"  Max:  {max(confidences):.3f}")
    print(f"  Mean: {sum(confidences)/len(confidences):.3f}")
    
    print(f"\nLow-confidence samples (< 0.6): {sum(1 for c in confidences if c < 0.6)} ({100*sum(1 for c in confidences if c < 0.6)/len(confidences):.1f}%)")
    print(f"  → Candidates for active learning / human review")
    
    print(f"\n✓ Done! All splits generated successfully.")


if __name__ == "__main__":
    main()
