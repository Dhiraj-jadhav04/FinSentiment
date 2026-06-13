import streamlit as st
import os
import database.db_manager as db
from app.utils import inject_custom_css

st.set_page_config(
    page_title="FinSentiment - LLM Financial Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database
db.init_db()

# Styling
inject_custom_css()

# Main page Content
st.markdown('<div class="gradient-header">FinSentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subheader">Next-Generation LLM Sentiment Intelligence</div>', unsafe_allow_html=True)

st.markdown("""
<div class="premium-card">
    <h3>Welcome to FinSentiment</h3>
    <p>FinSentiment is a complete, enterprise-grade AI solution powered by <strong>FinBERT</strong> (a specialized financial-domain BERT model). It classifies financial texts—such as company announcements, earnings reports, news articles, and customer support channels—into <strong>Positive</strong>, <strong>Negative</strong>, or <strong>Neutral</strong> sentiment. </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="premium-card" style="height: 350px;">
        <h4>Key System Capabilities</h4>
        <ul>
            <li><strong>FinBERT NLP Classifier</strong>: Pre-trained and fine-tuned on financial language to detect subtle market expressions.</li>
            <li><strong>Explainable AI (XAI)</strong>: Occlusion analysis shows word-by-word impact on sentiment confidence scores.</li>
            <li><strong>Batch Processing</strong>: Upload CSV files to generate predictions instantly on thousands of rows.</li>
            <li><strong>Real-Time Analytics</strong>: Historical sentiment trends, news feed simulations, and overall market signals.</li>
            <li><strong>Automated PDF Reporting</strong>: Export model stats, loss curves, and confusion matrices into formatted PDFs.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="premium-card" style="height: 350px; text-align: center;">
        <h4>NLP Model Architecture Workflow</h4>
        <div style="margin-top: 25px;">
            <div style="background: rgba(99,102,241,0.2); border: 1px solid #6366f1; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                <strong>1. Inputs</strong>: CSV Uploads, User Entry, RSS Feeds
            </div>
            <div style="background: rgba(56,189,248,0.2); border: 1px solid #38bdf8; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                <strong>2. Preprocessing</strong>: HTML Strip, URL Strip, Tokenization
            </div>
            <div style="background: rgba(16,185,129,0.2); border: 1px solid #10b981; border-radius: 8px; padding: 10px; margin-bottom: 10px;">
                <strong>3. FinBERT Model</strong>: Contextual embeddings & sequence classification
            </div>
            <div style="background: rgba(236,72,153,0.2); border: 1px solid #ec4899; border-radius: 8px; padding: 10px;">
                <strong>4. Output</strong>: Categorical probability distributions & word occlusion XAI
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.info("👈 Navigate through the application pages using the sidebar to upload datasets, fine-tune the model, run predictions, and explore analytics!")
