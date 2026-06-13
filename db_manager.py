import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "finsentiment.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for user predictions history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        confidence REAL NOT NULL,
        probabilities TEXT NOT NULL,
        source TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table for uploaded datasets
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table for model training metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        epochs INTEGER,
        batch_size INTEGER,
        learning_rate REAL,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        trained_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()

def save_prediction(text, sentiment, confidence, probabilities, source="Manual"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO predictions (text, sentiment, confidence, probabilities, source)
    VALUES (?, ?, ?, ?, ?)
    """, (text, sentiment, confidence, json.dumps(probabilities), source))
    conn.commit()
    conn.close()

def get_predictions(limit=100):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, text, sentiment, confidence, probabilities, source, timestamp 
    FROM predictions 
    ORDER BY timestamp DESC 
    LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        item = dict(r)
        item['probabilities'] = json.loads(item['probabilities'])
        result.append(item)
    return result

def save_dataset_meta(filename, row_count):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO datasets (filename, row_count)
    VALUES (?, ?)
    """, (filename, row_count))
    conn.commit()
    conn.close()

def get_datasets_meta():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_model_meta(model_name, epochs, batch_size, learning_rate, accuracy, precision, recall, f1):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO model_metadata (model_name, epochs, batch_size, learning_rate, accuracy, precision, recall, f1)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (model_name, epochs, batch_size, learning_rate, accuracy, precision, recall, f1))
    conn.commit()
    conn.close()

def get_latest_model_meta():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_metadata ORDER BY trained_at DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
