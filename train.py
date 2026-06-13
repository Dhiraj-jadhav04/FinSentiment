import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from training.preprocess import clean_text
import database.db_manager as db

MODEL_CARD = "ProsusAI/finbert"

class FinSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(data_path, epochs=3, batch_size=8, lr=2e-5, fast_mode=True, progress_callback=None):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df = df.dropna(subset=['text', 'sentiment'])
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Map sentiments to ids
    label_map = {"positive": 0, "negative": 1, "neutral": 2}
    df['label'] = df['sentiment'].str.lower().str.strip().map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    # Check shape
    if len(df) < 5:
        raise ValueError("Dataset too small. Please provide at least 5 samples.")

    # Train test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_texts = train_df['cleaned_text'].to_numpy()
    train_labels = train_df['label'].to_numpy()
    val_texts = val_df['cleaned_text'].to_numpy()
    val_labels = val_df['label'].to_numpy()
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CARD)
    
    # Datasets and loaders
    train_dataset = FinSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinSentimentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load model
    print(f"Loading base model {MODEL_CARD}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CARD)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Fast mode: freeze BERT backbone, only train the classifier head
    if fast_mode:
        print("Fast Mode Enabled: Freezing BERT backbone. Training classification head only.")
        for param in model.bert.parameters():
            param.requires_grad = False
            
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # History
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            if progress_callback:
                prog = ((epoch) / epochs) + ((idx / len(train_loader)) / epochs)
                progress_callback(prog, f"Epoch {epoch+1}/{epochs} - Batch {idx+1}/{len(train_loader)}")
                
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_losses.append(loss.item())
                
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_targets, val_preds)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    # Final metrics
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    
    precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='weighted', zero_division=0)
    
    # Save model
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "finbert_finetuned")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Create evaluation visualizations
    plot_eval_curves(history)
    plot_confusion_matrix(val_targets, val_preds)
    
    # Save model metadata to Database
    db.save_model_meta(
        model_name="FinBERT Fine-tuned",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        accuracy=float(val_acc),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1)
    )
    
    if progress_callback:
        progress_callback(1.0, "Training completed and model saved!")
        
    return {
        "accuracy": val_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "history": history
    }

def plot_eval_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', color='#FF4B4B', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', color='#1C83E1', marker='x')
    plt.title('Training & Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "loss_curves.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    labels = ["Positive", "Negative", "Neutral"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "confusion_matrix.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
