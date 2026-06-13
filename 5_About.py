import streamlit as st
from app.utils import inject_custom_css

inject_custom_css()

st.markdown('<div class="gradient-header">About the Project</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subheader">FinSentiment: Large Language Models for Financial NLP</div>', unsafe_allow_html=True)

st.markdown("""
<div class="premium-card">
    <h3>Project Objective</h3>
    <p>FinSentiment represents a comprehensive implementation of domain-specific sentiment classification, combining the raw power of Hugging Face transformer models with structured database record storage and dynamic, real-time analytics dashboards. By utilizing a pre-trained <strong>FinBERT</strong> backbone, the application accurately evaluates financial expressions that traditional lexicon models typically misclassify.</p>
</div>

<div class="premium-card">
    <h3>Deep Learning Core: FinBERT</h3>
    <p><strong>FinBERT</strong> is a BERT-based language model specifically pre-trained on a large financial corpus consisting of financial news, Reuters articles, earnings call transcripts, and corporate disclosures. The model is specifically fine-tuned for financial sentiment classification:</p>
    <ul>
        <li><strong>Language Context</strong>: While standard BERT reads "bullish" or "revenue beat" as general terms, FinBERT understands their direct positive financial impact.</li>
        <li><strong>Output Labels</strong>: Classifies sequences into Positive, Negative, or Neutral sentiments.</li>
        <li><strong>Confidence Scores</strong>: Extracts logits representing the probability distribution over classes, giving traders and analysts calibrated signals.</li>
    </ul>
</div>

<div class="premium-card">
    <h3>Technology Stack Summary</h3>
    <table style="width:100%; border-collapse: collapse; margin-top:10px;">
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align:left;">
            <th style="padding:10px; color:#818cf8;">Layer</th>
            <th style="padding:10px; color:#818cf8;">Framework / Library</th>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding:10px; font-weight:bold;">Deep Learning Core</td>
            <td style="padding:10px;">PyTorch, Hugging Face Transformers, FinBERT (ProsusAI)</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding:10px; font-weight:bold;">Data Processing</td>
            <td style="padding:10px;">Pandas, NumPy, Scikit-Learn</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding:10px; font-weight:bold;">Database Layer</td>
            <td style="padding:10px;">SQLite3 (History, Dataset Meta, Model Runs)</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding:10px; font-weight:bold;">Frontend Interface</td>
            <td style="padding:10px;">Streamlit, Plotly Express (Interactive charts)</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding:10px; font-weight:bold;">Visualizations</td>
            <td style="padding:10px;">Matplotlib, Seaborn, WordCloud</td>
        </tr>
        <tr>
            <td style="padding:10px; font-weight:bold;">Report Generation</td>
            <td style="padding:10px;">FPDF2 (PDF Document Export)</td>
        </tr>
    </table>
</div>

<div class="premium-card" style="text-align:center;">
    <p>© 2026 FinSentiment AI. Developed using state-of-the-art NLP models.</p>
</div>
""", unsafe_allow_html=True)
