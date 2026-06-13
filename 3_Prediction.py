import streamlit as st
import pandas as pd
import plotly.express as px
from inference.predictor import FinSentimentPredictor
import database.db_manager as db
from app.utils import inject_custom_css

inject_custom_css()

st.markdown('<div class="gradient-header">Sentiment Inference Console</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subheader">Analyze single phrases or upload batches for automated text classification</div>', unsafe_allow_html=True)

# Load predictor
@st.cache_resource
def get_predictor():
    return FinSentimentPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading FinBERT model: {e}. Please ensure correct pytorch installation.")
    st.stop()

tab1, tab2 = st.tabs(["Manual Text Analysis", "Batch File Classification"])

with tab1:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    user_input = st.text_area("Enter financial statement or announcement here:", 
                              value="Company reports record quarterly profits and increased revenue.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter text first.")
        else:
            with st.spinner("Analyzing text..."):
                label, confidence, probs = predictor.predict(user_input)
                
                # Save to database
                db.save_prediction(user_input, label, confidence, probs, source="Manual")
                
                # Format Output Banner
                sentiment_colors = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#F59E0B"}
                color = sentiment_colors.get(label, "#f8fafc")
                
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.7); border: 2px solid {color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 25px;">
                    <h2 style="color: {color}; margin: 0;">Sentiment: {label.upper()}</h2>
                    <h3 style="margin: 5px 0;">Confidence Score: {confidence*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Charts and XAI Layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Probability Distribution")
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
                    fig = px.bar(
                        prob_df, 
                        x="Category", 
                        y="Probability", 
                        color="Category",
                        color_discrete_map={"positive": "#10B981", "negative": "#EF4444", "neutral": "#F59E0B"},
                        range_y=[0, 1]
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.markdown("#### Explainable AI (XAI) - Word Importance")
                    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Word highlighting represents the contribution of each word to the predicted class. <span style='color:#10b981;font-weight:bold;'>Green</span> words boost confidence, while <span style='color:#ef4444;font-weight:bold;'>Red</span> words subtract from confidence.</p>", unsafe_allow_html=True)
                    
                    xai_scores = predictor.explain_prediction(user_input)
                    
                    # Generate highlighted HTML
                    html_spans = []
                    for item in xai_scores:
                        word = item['word']
                        ns = item['normalized_score']
                        
                        if ns > 0:
                            # Green gradient
                            bg = f"rgba(16, 185, 129, {ns * 0.4})"
                            border = f"1px solid rgba(16, 185, 129, {ns * 0.8})"
                        elif ns < 0:
                            # Red gradient
                            bg = f"rgba(239, 68, 68, {abs(ns) * 0.4})"
                            border = f"1px solid rgba(239, 68, 68, {abs(ns) * 0.8})"
                        else:
                            bg = "transparent"
                            border = "1px solid rgba(255, 255, 255, 0.1)"
                            
                        html_spans.append(f"<span style='background: {bg}; border: {border}; padding: 3px 6px; margin: 3px; border-radius: 4px; display: inline-block;'>{word}</span>")
                        
                    highlighted_html = f"<div style='background: rgba(30,41,59,0.7); padding: 15px; border-radius: 10px; line-height: 2.5; font-size:1.1rem; border:1px solid rgba(255,255,255,0.1);'>{' '.join(html_spans)}</div>"
                    st.markdown(highlighted_html, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    batch_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"], key="batch_upload")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        st.write(f"Preview of uploaded CSV ({len(batch_df)} rows):")
        st.dataframe(batch_df.head(5), use_container_width=True)
        
        text_column = st.selectbox("Choose Text Column for Sentiment Prediction", list(batch_df.columns), key="batch_col")
        
        if st.button("Run Batch Inference"):
            texts = batch_df[text_column].astype(str).tolist()
            progress = st.progress(0.0)
            status = st.empty()
            
            predictions = []
            confidences = []
            
            # Predict in batches to show progress
            batch_size = 16
            for idx in range(0, len(texts), batch_size):
                sub_texts = texts[idx : idx + batch_size]
                batch_res = predictor.predict_batch(sub_texts)
                
                for r in batch_res:
                    predictions.append(r['sentiment'])
                    confidences.append(r['confidence'])
                    
                    # Store prediction in database history
                    db.save_prediction(
                        text=texts[len(predictions)-1],
                        sentiment=r['sentiment'],
                        confidence=r['confidence'],
                        probabilities=r['probabilities'],
                        source=f"Batch ({batch_file.name})"
                    )
                    
                progress.progress(min(1.0, len(predictions) / len(texts)))
                status.text(f"Processed {len(predictions)}/{len(texts)} texts...")
                
            batch_df['predicted_sentiment'] = predictions
            batch_df['confidence_score'] = confidences
            
            status.success(f"Batch prediction completed! Processed {len(texts)} rows.")
            
            # Display stats
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### Batch Results")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Download predicted CSV
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Classified CSV Results",
                data=csv_data,
                file_name="classified_financial_sentiment.csv",
                mime="text/csv"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Plot distribution of batch
            counts = batch_df['predicted_sentiment'].value_counts().reset_index()
            counts.columns = ['sentiment', 'count']
            fig_batch = px.bar(
                counts, 
                x='sentiment', 
                y='count', 
                color='sentiment',
                color_discrete_map={"positive": "#10B981", "negative": "#EF4444", "neutral": "#F59E0B"},
                title="Batch Classification Sentiment Distribution"
            )
            fig_batch.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
            st.plotly_chart(fig_batch, use_container_width=True)
