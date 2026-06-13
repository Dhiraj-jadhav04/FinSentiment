import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\?\!\-\%\$]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(df, text_col='text', label_col='sentiment'):
    df = df.copy()
    # Handle missing values
    df = df.dropna(subset=[text_col])
    # Clean text
    df['cleaned_text'] = df[text_col].apply(clean_text)
    # Ensure date column exists or generate one
    if 'date' not in df.columns:
        df['date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    # Ensure source column exists
    if 'source' not in df.columns:
        df['source'] = 'Unknown'
    return df
