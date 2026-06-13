import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import database.db_manager as db
from app.utils import inject_custom_css, generate_wordcloud
from inference.predictor import FinSentimentPredictor
import time

inject_custom_css()

st.markdown('<div class="gradient-header">Sentiment Intelligence & Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subheader">Aggregated history, simulated real-time tickers, and market indicators</div>', unsafe_allow_html=True)

# Fetch data from DB
history = db.get_predictions(limit=500)

if not history:
    st.info("No prediction history found. Run manual or batch predictions in the 'Prediction Console' tab to generate analytics.")
else:
    df_hist = pd.DataFrame(history)
    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
    
    # 1. Summary KPIs
    total_preds = len(df_hist)
    avg_conf = df_hist['confidence'].mean()
    pos_count = len(df_hist[df_hist['sentiment'] == 'positive'])
    neg_count = len(df_hist[df_hist['sentiment'] == 'negative'])
    
    st.markdown("### Cumulative Analytics Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Classified Phrases", f"{total_preds}")
    kpi2.metric("Average Confidence", f"{avg_conf*100:.1f}%")
    kpi3.metric("Positive Assertions", f"{pos_count}")
    kpi4.metric("Negative Assertions", f"{neg_count}")
    
    # Gauge Chart - Investment Sentiment Indicator
    st.markdown("---")
    col_g1, col_g2 = st.columns([1, 2])
    
    with col_g1:
        st.markdown("#### Market Sentiment Gauge")
        # Compute sentiment index: (Positive - Negative) / Total * 100 -> maps between -100 (Bearish) and +100 (Bullish)
        sent_index = 0
        if total_preds > 0:
            sent_index = ((pos_count - neg_count) / total_preds) * 100
            
        # Plotly Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sent_index,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Bullish vs Bearish Indicator", 'font': {'color': '#f8fafc'}},
            gauge = {
                'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                'bar': {'color': "#6366f1"},
                'bgcolor': "rgba(30, 41, 59, 0.7)",
                'borderwidth': 2,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [-100, -30], 'color': '#ef4444'},
                    {'range': [-30, 30], 'color': '#6b7280'},
                    {'range': [30, 100], 'color': '#10b981'}
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc", 'family': "Outfit"})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col_g2:
        st.markdown("#### Sentiment Trend Forecasting")
        # Resample data by date
        df_hist['date_only'] = df_hist['timestamp'].dt.date
        date_counts = df_hist.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0).reset_index()
        
        # Calculate positive moving average / ratio
        for col in ['positive', 'negative', 'neutral']:
            if col not in date_counts.columns:
                date_counts[col] = 0
                
        date_counts['Bullish Ratio'] = date_counts['positive'] / (date_counts['positive'] + date_counts['negative'] + date_counts['neutral']) * 100
        
        fig_line = px.line(
            date_counts, 
            x='date_only', 
            y=['positive', 'negative', 'neutral'],
            labels={'value': 'Count', 'date_only': 'Date'},
            title="Classification Trends Over Time",
            color_discrete_map={"positive": "#10B981", "negative": "#EF4444", "neutral": "#F59E0B"}
        )
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig_line, use_container_width=True)

    # 2. Source Comparison & Word Clouds
    st.markdown("---")
    col_sc1, col_sc2 = st.columns(2)
    
    with col_sc1:
        st.markdown("#### Sentiment Share by News Source")
        source_df = df_hist.groupby(['source', 'sentiment']).size().unstack(fill_value=0).reset_index()
        for col in ['positive', 'negative', 'neutral']:
            if col not in source_df.columns:
                source_df[col] = 0
                
        fig_source = px.bar(
            source_df, 
            x='source', 
            y=['positive', 'negative', 'neutral'],
            title="Sentiment Breakdown by Channel / Source",
            color_discrete_map={"positive": "#10B981", "negative": "#EF4444", "neutral": "#F59E0B"},
            barmode='stack'
        )
        fig_source.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig_source, use_container_width=True)
        
    with col_sc2:
        st.markdown("#### Cloud of Key Sentiment Terms")
        # Generate word cloud from all historical texts
        all_texts = df_hist['text'].tolist()
        wc_img = generate_wordcloud(all_texts)
        st.image(wc_img, use_container_width=True)

# 3. Simulated Real-time Monitor
st.markdown("---")
st.markdown("### Real-Time Financial News Monitor")
st.markdown("<p style='color:#94a3b8;'>Simulates fetching live headlines and classifying sentiment instantly in a streaming fashion.</p>", unsafe_allow_html=True)

simulated_headlines = [
    {"text": "Federal Reserve signals potential rate cut by end of third quarter.", "source": "Bloomberg"},
    {"text": "Retail sales drop unexpectedly as inflation weighs heavily on consumers.", "source": "Reuters"},
    {"text": "Microchip manufacturer secures $10B government subsidy for new Arizona plant.", "source": "TechCrunch"},
    {"text": "Automotive giant faces supply chain hurdles due to port strike.", "source": "CNBC"},
    {"text": "E-commerce platform surpasses 500 million active users globally.", "source": "MarketWatch"},
    {"text": "Oil prices surge to 6-month high amid rising geopolitical tensions.", "source": "Reuters"},
    {"text": "Software provider reports wider than expected loss in Q3 report.", "source": "Yahoo Finance"}
]

@st.cache_resource
def get_live_predictor():
    return FinSentimentPredictor()

if st.button("Start Live News Monitoring Feed"):
    live_predictor = get_live_predictor()
    feed_placeholder = st.empty()
    
    for headline in simulated_headlines:
        label, confidence, _ = live_predictor.predict(headline['text'])
        
        # Save to database
        db.save_prediction(headline['text'], label, confidence, {"positive": 0.0, "negative": 0.0, "neutral": 0.0}, source=f"Live ({headline['source']})")
        
        sentiment_badge = ""
        if label == "positive":
            sentiment_badge = "<span style='background:#10b981; color:white; padding:3px 8px; border-radius:4px; font-weight:bold;'>Positive</span>"
        elif label == "negative":
            sentiment_badge = "<span style='background:#ef4444; color:white; padding:3px 8px; border-radius:4px; font-weight:bold;'>Negative</span>"
        else:
            sentiment_badge = "<span style='background:#6b7280; color:white; padding:3px 8px; border-radius:4px; font-weight:bold;'>Neutral</span>"
            
        with feed_placeholder.container():
            st.markdown(f"""
            <div class="premium-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#818cf8; font-weight:bold;">Source: {headline['source']}</span>
                    {sentiment_badge}
                </div>
                <h4 style="margin: 10px 0;">{headline['text']}</h4>
                <div style="font-size:0.8rem; color:#94a3b8;">Confidence: {confidence*100:.1f}% | Processed: Just now</div>
            </div>
            """, unsafe_allow_html=True)
            
        time.sleep(2)
        
    st.success("Simulation finished! Refresh to stream again.")
