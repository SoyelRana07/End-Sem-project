"""
Optimized BERT Text Classifier with batching support
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config
import numpy as np

class OptimizedCaptionClassifier:
    def __init__(self, model_path=config.BERT_MODEL_PATH):
        """Initialize BERT classifier with GPU support"""
        print(f"Loading BERT classifier from: {model_path}")
        
        # Set device
        self.device = "cuda" if config.ENABLE_GPU and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable half precision on GPU
        if self.device == "cuda":
            self.model = self.model.half()
        
        # Load label mapping
        self.id2label = self.model.config.id2label
        
        print(f"BERT model loaded with {len(self.id2label)} classes")
    
    @torch.no_grad()
    def classify(self, caption):
        """
        Classify a single caption
        
        Args:
            caption: string
            
        Returns:
            dict: {label, confidence, all_scores}
        """
        # Tokenize
        inputs = self.tokenizer(
            caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.BERT_MAX_LENGTH
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
        
        # Convert to CPU and numpy
        confidence = confidence.item()
        predicted_class = predicted_class.item()
        all_probs = probs.cpu().numpy()[0]
        
        # Get label
        label = self.id2label[predicted_class]
        
        # Get all scores
        all_scores = {
            self.id2label[i]: float(all_probs[i])
            for i in range(len(all_probs))
        }
        
        return {
            "label": label,
            "confidence": confidence,
            "all_scores": all_scores
        }
    
    @torch.no_grad()
    def classify_batch(self, captions):
        """
        Classify multiple captions at once (more efficient)
        
        Args:
            captions: list of strings
            
        Returns:
            list of dicts
        """
        if not captions:
            return []
        
        # Tokenize all captions
        inputs = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.BERT_MAX_LENGTH
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidences, predicted_classes = torch.max(probs, dim=-1)
        
        # Convert to CPU
        confidences = confidences.cpu().numpy()
        predicted_classes = predicted_classes.cpu().numpy()
        all_probs = probs.cpu().numpy()
        
        # Format results
        results = []
        for i in range(len(captions)):
            label = self.id2label[predicted_classes[i]]
            confidence = float(confidences[i])
            all_scores = {
                self.id2label[j]: float(all_probs[i][j])
                for j in range(len(all_probs[i]))
            }
            
            results.append({
                "label": label,
                "confidence": confidence,
                "all_scores": all_scores
            })
        
        return results
