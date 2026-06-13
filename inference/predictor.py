import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Default FinBERT model card
MODEL_CARD = "ProsusAI/finbert"

class FinSentimentPredictor:
    def __init__(self, model_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine paths
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "finbert_finetuned")
        
        # Load custom model if exists and directory is provided
        if model_dir and os.path.exists(model_dir):
            self.model_path = model_dir
        elif os.path.exists(local_model_path):
            self.model_path = local_model_path
        else:
            self.model_path = MODEL_CARD
            
        print(f"Loading tokenizer and model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Label map based on ProsusAI/finbert configuration
        # 0: positive, 1: negative, 2: neutral
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
        self.label2id = {"positive": 0, "negative": 1, "neutral": 2}

    def predict(self, text):
        if not text.strip():
            return "neutral", 1.0, {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
            
        pred_idx = probs.argmax().item()
        label = self.id2label[pred_idx]
        confidence = probs[pred_idx]
        
        prob_dist = {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2])
        }
        
        return label, confidence, prob_dist

    def predict_batch(self, texts, batch_size=16):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                
            for p in probs:
                p_idx = p.argmax().item()
                results.append({
                    "sentiment": self.id2label[p_idx],
                    "confidence": float(p[p_idx]),
                    "probabilities": {
                        "positive": float(p[0]),
                        "negative": float(p[1]),
                        "neutral": float(p[2])
                    }
                })
        return results

    def explain_prediction(self, text):
        # Occlusion-based feature importance method
        words = text.split()
        if len(words) <= 1:
            return [{"word": w, "score": 0.0} for w in words]
            
        # Get baseline prediction
        base_label, base_conf, base_probs = self.predict(text)
        base_class_idx = self.label2id[base_label]
        base_class_prob = base_probs[base_label]
        
        word_scores = []
        for i in range(len(words)):
            # Reconstruct sentence omitting the current word
            perturbed_words = words[:i] + words[i+1:]
            perturbed_text = " ".join(perturbed_words)
            
            # Predict on perturbed text
            _, _, pert_probs = self.predict(perturbed_text)
            pert_class_prob = pert_probs[base_label]
            
            # The drop in probability of the predicted class indicates importance
            importance = base_class_prob - pert_class_prob
            word_scores.append({
                "word": words[i],
                "score": float(importance)
            })
            
        # Normalize scores to lie between -1.0 and 1.0 for visualization
        max_abs = max(abs(x['score']) for x in word_scores) if word_scores else 0
        if max_abs > 0:
            for score in word_scores:
                score['normalized_score'] = score['score'] / max_abs
        else:
            for score in word_scores:
                score['normalized_score'] = 0.0
                
        return word_scores
