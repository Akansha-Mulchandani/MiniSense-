"""
Generate labels for survey responses using rule-based logic.
No API calls needed - fully local.
"""
import json
import argparse
from collections import Counter

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

# Keyword mapping
KEYWORD_MAP = {
    "Food Quality": ["food", "menu", "taste", "quality", "meal", "delicious", "bland", "fresh", "stale"],
    "Wait Time": ["wait", "slow", "time", "queue", "long", "delay", "fast", "quick"],
    "Staff": ["staff", "service", "rude", "friendly", "helpful", "attitude", "polite"],
    "Ambiance": ["ambiance", "clean", "noise", "atmosphere", "music", "loud", "quiet", "dirty", "tidy"],
}

def classify_response(rating: int, free_text: str) -> tuple[str, int]:
    """
    Classify a response into one of 8 categories.
    Returns (label, label_id)
    """
    text_lower = free_text.lower()
    
    # Determine sentiment from rating
    if rating >= 4:
        sentiment = "Positive"
    elif rating <= 2:
        sentiment = "Negative"
    else:
        # Rating 3: use keyword fallback to determine sentiment
        # Count positive vs negative keywords
        positive_keywords = ["great", "good", "excellent", "love", "best", "amazing", "fantastic", "perfect"]
        negative_keywords = ["bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing"]
        
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        else:
            # Default to Positive for neutral
            sentiment = "Positive"
    
    # Determine aspect from keywords
    aspect_scores = {}
    for aspect, keywords in KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            aspect_scores[aspect] = score
    
    # Default to highest frequency aspect or Staff as fallback
    if aspect_scores:
        aspect = max(aspect_scores, key=aspect_scores.get)
    else:
        aspect = "Staff"  # Default fallback
    
    label = f"{sentiment} – {aspect}"
    label_id = LABEL_MAP[label]
    
    return label, label_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regeneration even if labeled data exists")
    args = parser.parse_args()
    
    # Check if labeled data already exists
    labeled_path = "data/labeled_responses.jsonl"
    if not args.force:
        try:
            with open(labeled_path, "r") as f:
                if f.read().strip():
                    print(f"Labeled data already exists at {labeled_path}. Use --force to regenerate.")
                    return
        except FileNotFoundError:
            pass
    
    # Load survey responses
    print("Loading survey responses...")
    with open("data/survey_responses.json", "r") as f:
        data = json.load(f)
        responses = data["responses"]
    
    print(f"Processing {len(responses)} responses...")
    
    # Classify each response
    labeled_data = []
    label_counts = Counter()
    
    for r in responses:
        label, label_id = classify_response(r["rating"], r["free_text"])
        labeled_data.append({
            "input_text": r["free_text"],
            "label": label,
            "label_id": label_id,
        })
        label_counts[label] += 1
    
    # Save labeled data as JSONL
    print(f"Saving labeled data to {labeled_path}...")
    with open(labeled_path, "w") as f:
        for item in labeled_data:
            f.write(json.dumps(item) + "\n")
    
    # Print label distribution
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    # Save column_map.yaml
    column_map_path = "data/column_map.yaml"
    print(f"\nSaving column map to {column_map_path}...")
    with open(column_map_path, "w") as f:
        f.write("input_field: free_text\n")
        f.write("label_field: label\n")
        f.write("label_map:\n")
        for label, label_id in LABEL_MAP.items():
            f.write(f'  "{label}": {label_id}\n')
    
    print(f"\nDone! Generated {len(labeled_data)} labeled responses.")

if __name__ == "__main__":
    main()
