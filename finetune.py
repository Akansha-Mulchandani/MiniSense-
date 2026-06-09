"""
Fine-tune distilbert-base-uncased on labeled survey responses.
Fully local - no API calls needed.
"""
import json
import yaml
import os
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import torch

# Load column map
with open("data/column_map.yaml", "r") as f:
    column_map = yaml.safe_load(f)

LABEL_MAP = column_map["label_map"]
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

def load_labeled_data():
    """Load labeled responses from JSONL."""
    data = []
    with open("data/labeled_responses.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def stratified_sample(data, samples_per_class=300):
    """Take a stratified sample ensuring equal samples per class."""
    # Group by label
    label_groups = {}
    for item in data:
        label = item["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    # Sample from each group
    sampled = []
    for label, items in label_groups.items():
        if len(items) >= samples_per_class:
            sampled.extend(np.random.choice(items, samples_per_class, replace=False))
        else:
            # If not enough samples, take all and oversample
            sampled.extend(items)
            needed = samples_per_class - len(items)
            sampled.extend(np.random.choice(items, needed, replace=True))
    
    return sampled

def tokenize_function(examples, tokenizer):
    """Tokenize the input texts."""
    return tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    macro_f1 = f1_score(labels, preds, average="macro")
    per_class_f1 = f1_score(labels, preds, average=None)
    
    return {
        "macro_f1": macro_f1,
        **{f"f1_class_{i}": score for i, score in enumerate(per_class_f1)},
    }

def main():
    print("Loading labeled data...")
    data = load_labeled_data()
    print(f"Total labeled samples: {len(data)}")
    
    # Print label distribution
    label_counts = Counter(item["label"] for item in data)
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    # Stratified sample
    print("\nTaking stratified sample (300 per class)...")
    sampled = stratified_sample(data, samples_per_class=300)
    print(f"Sampled {len(sampled)} records")
    
    # Split 80/10/10 train/val/test
    print("\nSplitting data 80/10/10 train/val/test...")
    train_data, temp_data = train_test_split(
        sampled, test_size=0.2, stratify=[item["label_id"] for item in sampled], random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, stratify=[item["label_id"] for item in temp_data], random_state=42
    )
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Load tokenizer and model
    print("\nLoading distilbert-base-uncased...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True
    )
    
    # Remove unused columns and rename label_id to labels
    train_dataset = train_dataset.remove_columns(["input_text", "label"])
    val_dataset = val_dataset.remove_columns(["input_text", "label"])
    test_dataset = test_dataset.remove_columns(["input_text", "label"])
    
    train_dataset = train_dataset.rename_column("label_id", "labels")
    val_dataset = val_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")
    
    # Set format for PyTorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    # Training arguments
    output_dir = "models/classifier"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        report_to="none",  # Disable wandb/mlflow
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    print("\n=== Test Set Results ===")
    print(f"Macro F1: {test_results['eval_macro_f1']:.4f}")
    
    # Print per-class F1
    print("\nPer-class F1:")
    for i in range(NUM_LABELS):
        label_name = ID_TO_LABEL[i]
        f1_score = test_results.get(f"eval_f1_class_{i}", 0.0)
        print(f"  {label_name}: {f1_score:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label map for inference
    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    print("\nFine-tuning complete!")

if __name__ == "__main__":
    main()
