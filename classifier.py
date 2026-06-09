"""
Classifier inference wrapper for fine-tuned survey response classifier.
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class Classifier:
    def __init__(self, model_path: str = "models/classifier/"):
        """
        Load the fine-tuned classifier from the specified path.
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run finetune.py first to train the model."
            )
        
        # Load label map
        label_map_path = os.path.join(model_path, "label_map.json")
        with open(label_map_path, "r") as f:
            self.label_map = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Load tokenizer and model
        print(f"Loading classifier from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Classifier loaded successfully on {self.device}")
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict the label for a single text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with 'label' and 'confidence' keys
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
        
        label = self.id_to_label[pred_id]
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict labels for multiple texts in batch.
        
        Args:
            texts: List of input texts to classify
            
        Returns:
            List of dictionaries with 'label' and 'confidence' keys
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict batch
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).tolist()
            confidences = probs[torch.arange(len(texts)), pred_ids].tolist()
        
        results = []
        for pred_id, confidence in zip(pred_ids, confidences):
            label = self.id_to_label[pred_id]
            results.append({
                "label": label,
                "confidence": round(confidence, 4),
            })
        
        return results


if __name__ == "__main__":
    # Test the classifier
    try:
        classifier = Classifier()
        
        # Test single prediction
        test_text = "The wait was too long and staff was rude"
        result = classifier.predict(test_text)
        print(f"\nTest prediction:")
        print(f"Text: {test_text}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']}")
        
        # Test batch prediction
        test_texts = [
            "Great food and excellent service!",
            "The wait time was terrible.",
            "Clean atmosphere but slow service.",
        ]
        results = classifier.predict_batch(test_texts)
        print(f"\nBatch predictions:")
        for text, result in zip(test_texts, results):
            print(f"Text: {text}")
            print(f"Label: {result['label']}, Confidence: {result['confidence']}")
            print()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run finetune.py first to train the model.")
