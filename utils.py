import os
import streamlit as st
from fpdf import FPDF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background & Text Color */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* Card Component */
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    /* Header Gradients */
    .gradient-header {
        background: linear-gradient(to right, #818cf8, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .gradient-subheader {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    
    /* Buttons Customization */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)

class FinSentimentReportPDF(FPDF):
    def header(self):
        # Draw header banner
        self.set_fill_color(15, 23, 42)  # #0F172A
        self.rect(0, 0, 210, 35, 'F')
        
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 20)
        self.cell(0, 15, "FINSENTIMENT - PERFORMANCE REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "I", 10)
        self.cell(0, 5, "LLM-Based Financial Sentiment Analyzer", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_text_color(100, 100, 100)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(metrics, loss_curves_path=None, cm_path=None):
    pdf = FinSentimentReportPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    
    # Report Info
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, "This report documents the performance evaluation of the fine-tuned FinBERT model. The model is optimized for predicting Positive, Negative, and Neutral sentiments in financial texts (news, earning announcements, customer communications).")
    pdf.ln(5)
    
    # Performance Metrics Table
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "2. Evaluation Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Table Header
    pdf.set_fill_color(99, 102, 241)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("helvetica", "B", 11)
    pdf.cell(45, 8, "Metric", border=1, fill=True)
    pdf.cell(45, 8, "Score", border=1, fill=True)
    pdf.ln()
    
    # Table Rows
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("helvetica", "", 11)
    for m, val in metrics.items():
        pdf.cell(45, 8, m.capitalize(), border=1)
        pdf.cell(45, 8, f"{val*100:.2f}%", border=1)
        pdf.ln()
    pdf.ln(10)
    
    # Visualizations Page
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "3. Training Visualizations", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    if loss_curves_path and os.path.exists(loss_curves_path):
        pdf.cell(0, 10, "Training Loss Curves", new_x="LMARGIN", new_y="NEXT")
        pdf.image(loss_curves_path, x=15, w=150, h=75)
        pdf.ln(10)
        
    if cm_path and os.path.exists(cm_path):
        pdf.add_page()
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, "4. Confusion Matrix", new_x="LMARGIN", new_y="NEXT")
        pdf.image(cm_path, x=20, w=130, h=100)
        
    return pdf.output()

def generate_wordcloud(texts):
    text_blob = " ".join(texts)
    if not text_blob.strip():
        text_blob = "No data available"
    wc = WordCloud(width=800, height=400, background_color='#0f172a', colormap='cool').generate(text_blob)
    
    # Convert Plot to Image Bytes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf
