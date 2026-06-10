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

def load_split_data(split_name="train"):
    """Load pre-split labeled responses from JSONL.
    
    Args:
        split_name: One of 'train', 'val', 'test'
    """
    data = []
    path = f"data/labeled_responses_{split_name}.jsonl"
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_f1(predictions, labels):
    """Simple F1 computation without sklearn."""
    # Macro F1: average F1 per class
    class_f1_scores = {}
    
    # Get unique labels
    unique_labels = set(labels)
    
    for label_id in unique_labels:
        # True positives, false positives, false negatives for this class
        tp = sum(1 for p, l in zip(predictions, labels) if p == label_id and l == label_id)
        fp = sum(1 for p, l in zip(predictions, labels) if p == label_id and l != label_id)
        fn = sum(1 for p, l in zip(predictions, labels) if p != label_id and l == label_id)
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 for this class
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        class_f1_scores[label_id] = f1
    
    # Macro F1: average across all classes
    macro_f1 = np.mean(list(class_f1_scores.values())) if class_f1_scores else 0.0
    
    return macro_f1, class_f1_scores

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
    
    macro_f1, class_f1_scores = compute_f1(preds.tolist(), labels.tolist())
    
    return {
        "macro_f1": macro_f1,
        **{f"f1_class_{i}": score for i, score in class_f1_scores.items()},
    }

def main():
    print("Loading pre-split labeled data...")
    train_data = load_split_data("train")
    val_data = load_split_data("val")
    test_data = load_split_data("test")
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print(f"Total samples: {len(train_data) + len(val_data) + len(test_data)}")
    
    # Print label distribution
    print("\nTrain set label distribution:")
    label_counts = Counter(item["label"] for item in train_data)
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
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
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",  # Disable wandb/mlflow
        fp16=True,  # Mixed precision - 2x faster
        optim="adamw_torch_fused",  # Fused optimizer
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
    
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print(f"Macro F1: {test_results['eval_macro_f1']:.4f}")
    
    # Print per-class F1
    print("\nPer-class F1:")
    for i in range(NUM_LABELS):
        label_name = ID_TO_LABEL[i]
        f1 = test_results.get(f"eval_f1_class_{i}", 0.0)
        print(f"  {label_name:30s}: {f1:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label map for inference
    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump(LABEL_MAP, f, indent=2)
    
    print("\n✓ Fine-tuning complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Ready to test with: python main.py --serve")

if __name__ == "__main__":
    main()
