"""
Generates 100,000 fake survey responses and saves to data/survey_responses.json
"""
import json, random, os
from faker import Faker
from datetime import date, timedelta

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

POSITIVE_TEMPLATES = [
    "Really enjoyed my visit. {} was excellent!",
    "Great experience overall. {} exceeded my expectations.",
    "The {} is top-notch. Will definitely return.",
    "Staff were friendly and {} was fantastic.",
    "Loved the atmosphere. {} was spot on.",
]
NEGATIVE_TEMPLATES = [
    "Very disappointed with {}. Won't be back.",
    "The {} was terrible. Needs serious improvement.",
    "Had a bad experience. {} was unacceptable.",
    "Not worth it — {} was way below standard.",
    "Frustrated with {}. Expected much better.",
]
NEUTRAL_TEMPLATES = [
    "The {} was okay. Nothing special.",
    "Average experience. {} could be better.",
    "It was fine. {} was acceptable.",
    "Decent visit. {} was neither great nor bad.",
]
ASPECTS = ["food quality", "wait time", "service", "cleanliness", "pricing", "staff attitude", "ambiance", "value for money"]

def gen_text(rating):
    aspect = random.choice(ASPECTS)
    if rating >= 4:
        tpl = random.choice(POSITIVE_TEMPLATES)
        if rating == 5:
            extra = " Highly recommend!"
        else:
            extra = ""
        return tpl.format(aspect) + extra
    elif rating <= 2:
        tpl = random.choice(NEGATIVE_TEMPLATES)
        return tpl.format(aspect)
    else:
        tpl = random.choice(NEUTRAL_TEMPLATES)
        return tpl.format(aspect)

def gen_date(month_offset=0):
    base = date(2026, 4, 1) if month_offset == 0 else date(2026, 5, 1)
    days = 30
    return (base + timedelta(days=random.randint(0, days - 1))).isoformat()

os.makedirs("data", exist_ok=True)
responses = []
for i in range(100000):
    biz = random.choice(BUSINESSES)
    survey = random.choice(SURVEYS)
    rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
    month_offset = random.randint(0, 1)
    responses.append({
        "response_id": f"r{i+1:06d}",
        "date": gen_date(month_offset),
        "business_id": biz["id"],
        "business_name": biz["name"],
        "survey_id": survey["id"],
        "survey_name": survey["name"],
        "rating": rating,
        "csat": rating,
        "response_channel": random.choice(CHANNELS),
        "free_text": gen_text(rating),
    })

with open("data/survey_responses.json", "w") as f:
    json.dump({"responses": responses}, f)

print(f"Generated {len(responses)} records -> data/survey_responses.json")
