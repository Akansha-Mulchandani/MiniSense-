"""
Generates 100,000 fake survey responses with stratified class distribution.
Ensures balanced representation across 8 sentiment+topic classes.

Data Strategy:
- 100k total responses divided into 8 classes (12,500 per class)
- Classes: Positive/Negative × (Food Quality, Wait Time, Staff, Ambiance)
- Stratification ensures equal class representation for unbiased training
- Dates span April-May 2026 (2 months)
- 3 businesses, 3 survey types, 4 response channels mixed
"""
import json, random, os
from faker import Faker
from datetime import date, timedelta
from collections import Counter

fake = Faker()
random.seed(42)

BUSINESSES = [
    {"id": "b01", "name": "GreenLeaf Bistro"},
    {"id": "b02", "name": "QuickFit Gym"},
    {"id": "b03", "name": "TechHub Coworking"},
]
SURVEYS = [
    {"id": "s01", "name": "Membership Value"},
    {"id": "s02", "name": "Food Quality"},
    {"id": "s03", "name": "Staff Experience"},
]
CHANNELS = ["mobile", "web", "kiosk", "email"]

# Stratified aspect+sentiment pairs for balanced class generation
# Each class will have exactly 12,500 samples (100k / 8 classes)
STRATIFIED_CLASSES = [
    ("food quality", "positive"),     # 0
    ("food quality", "negative"),     # 1
    ("wait time", "positive"),        # 2
    ("wait time", "negative"),        # 3
    ("staff", "positive"),            # 4
    ("staff", "negative"),            # 5
    ("ambiance", "positive"),         # 6
    ("ambiance", "negative"),         # 7
]

# Map sentiment to rating range for consistency
SENTIMENT_TO_RATING = {
    "positive": [4, 5],      # Positive: ratings 4-5
    "negative": [1, 2],      # Negative: ratings 1-2
    "neutral": [3],          # Neutral: rating 3
}

# Aspect-specific templates
TEMPLATES = {
    "food quality": {
        "positive": [
            "The food quality was excellent! Really enjoyed my visit.",
            "Best meal I've had. The food exceeded my expectations.",
            "The food quality is top-notch. Will definitely return.",
            "Fantastic food. Fresh and delicious!",
            "Loved the food. Highly recommend!",
        ],
        "negative": [
            "Very disappointed with the food quality. Won't be back.",
            "The food was terrible. Needs serious improvement.",
            "Had a bad experience with food. It was unacceptable.",
            "Not worth it — food was way below standard.",
            "Frustrated with food quality. Expected much better.",
        ],
    },
    "wait time": {
        "positive": [
            "Great service! The wait time was minimal.",
            "Impressed with how quickly I was served. Great experience!",
            "The wait time was perfect. Very efficient.",
            "Fast service. No complaints here!",
            "Best wait time I've experienced. Highly recommend!",
        ],
        "negative": [
            "Very disappointed with the wait time. Won't be back.",
            "The wait was terrible. Way too long.",
            "Had a bad experience. The wait was unacceptable.",
            "Not worth it — the wait time was way too long.",
            "Frustrated with long wait times. Expected better.",
        ],
    },
    "staff": {
        "positive": [
            "Staff were fantastic and very helpful!",
            "Great experience overall. The staff was excellent.",
            "Staff service is top-notch. Will definitely return.",
            "The staff were friendly and polite!",
            "Outstanding staff. Highly recommend!",
        ],
        "negative": [
            "Very disappointed with staff service. Won't be back.",
            "The staff were terrible. Needs serious improvement.",
            "Had a bad experience. Staff was rude and unhelpful.",
            "Not worth it — staff attitude was unacceptable.",
            "Frustrated with staff behavior. Expected much better.",
        ],
    },
    "ambiance": {
        "positive": [
            "Loved the atmosphere and ambiance!",
            "Great experience overall. The ambiance was excellent.",
            "The ambiance is top-notch. Very clean and inviting.",
            "Perfect atmosphere with great music and lighting!",
            "Excellent ambiance. Highly recommend!",
        ],
        "negative": [
            "Very disappointed with the ambiance. Won't be back.",
            "The atmosphere was terrible. Too loud and dirty.",
            "Had a bad experience. The ambiance was unacceptable.",
            "Not worth it — the place was too dirty and noisy.",
            "Frustrated with poor ambiance. Expected better.",
        ],
    },
}

def gen_text(aspect, sentiment):
    """Generate realistic free-text review for given aspect and sentiment."""
    return random.choice(TEMPLATES[aspect][sentiment])

def gen_rating(sentiment):
    """Generate rating consistent with sentiment."""
    return random.choice(SENTIMENT_TO_RATING[sentiment])

def gen_date(month_offset=0):
    """Generate date spanning April-May 2026."""
    base = date(2026, 4, 1) if month_offset == 0 else date(2026, 5, 1)
    days = 30
    return (base + timedelta(days=random.randint(0, days - 1))).isoformat()

os.makedirs("data", exist_ok=True)

# Generate stratified data: exactly 12,500 samples per class
responses = []
samples_per_class = 100000 // len(STRATIFIED_CLASSES)  # 12,500

print("Generating stratified survey data...")
print(f"Target: {samples_per_class} samples per class × {len(STRATIFIED_CLASSES)} classes = {samples_per_class * len(STRATIFIED_CLASSES)} total")

for class_idx, (aspect, sentiment) in enumerate(STRATIFIED_CLASSES):
    for sample_idx in range(samples_per_class):
        global_idx = class_idx * samples_per_class + sample_idx
        biz = random.choice(BUSINESSES)
        survey = random.choice(SURVEYS)
        rating = gen_rating(sentiment)
        month_offset = random.randint(0, 1)
        
        responses.append({
            "response_id": f"r{global_idx+1:06d}",
            "date": gen_date(month_offset),
            "business_id": biz["id"],
            "business_name": biz["name"],
            "survey_id": survey["id"],
            "survey_name": survey["name"],
            "rating": rating,
            "csat": rating,
            "response_channel": random.choice(CHANNELS),
            "free_text": gen_text(aspect, sentiment),
        })

# Shuffle to mix classes
random.shuffle(responses)

# Verify class distribution
from collections import Counter
class_counts = Counter()
for r in responses:
    text = r["free_text"].lower()
    rating = r["rating"]
    if rating >= 4:
        sentiment = "positive"
    elif rating <= 2:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    class_counts[sentiment] += 1

print(f"\nGenerated {len(responses)} records")
print("Sentiment distribution:")
for sentiment, count in class_counts.most_common():
    print(f"  {sentiment}: {count}")

with open("data/survey_responses.json", "w") as f:
    json.dump({"responses": responses}, f)

print(f"\n✓ Saved to data/survey_responses.json")
