import streamlit as st
import pandas as pd
import os
import plotly.express as px
from training.preprocess import preprocess_dataset
import database.db_manager as db
from app.utils import inject_custom_css

inject_custom_css()

st.markdown('<div class="gradient-header">Dataset Upload & Preprocessing</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subheader">Prepare financial news or customer communications for sentiment modeling</div>', unsafe_allow_html=True)

# Dataset selection source
source_option = st.radio("Choose Dataset Source", ["Use Built-in Sample Dataset", "Upload Custom CSV File"])

df = None
filename = ""

if source_option == "Use Built-in Sample Dataset":
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "sample_financial_data.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        filename = "sample_financial_data.csv"
        st.success(f"Successfully loaded sample dataset ({len(df)} rows)")
    else:
        st.error("Sample dataset not found. Running the launcher will recreate it.")
else:
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        filename = uploaded_file.name
        st.success(f"Successfully uploaded {filename} ({len(df)} rows)")

if df is not None:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Check for expected columns
    expected_cols = ['text', 'sentiment']
    cols_exist = all(col in df.columns for col in expected_cols)
    
    if not cols_exist:
        st.warning(f"Warning: Your CSV should ideally contain columns named: {expected_cols}. Found columns: {list(df.columns)}")
        text_col = st.selectbox("Select Text Column", list(df.columns))
        sentiment_col = st.selectbox("Select Sentiment/Target Column", ['None'] + list(df.columns))
    else:
        text_col = 'text'
        sentiment_col = 'sentiment'
        st.info("Found standard 'text' and 'sentiment' columns!")
        
    if st.button("Preprocess and Clean Dataset"):
        with st.spinner("Cleaning text, removing URLs and HTML tags..."):
            # If sentiment col is selected
            s_col = None if sentiment_col == 'None' else sentiment_col
            cleaned_df = preprocess_dataset(df, text_col=text_col, label_col=s_col)
            
            # Save data to state for page 2 (training)
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['dataset_filename'] = filename
            
            # Save metadata to DB
            db.save_dataset_meta(filename, len(cleaned_df))
            
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### Cleaned Data Preview")
            st.dataframe(cleaned_df[['text', 'cleaned_text', 'sentiment', 'source', 'date']].head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Visualizations of uploaded dataset
            if 'sentiment' in cleaned_df.columns:
                col1, col2 = st.columns(2)
                
                sentiment_counts = cleaned_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                with col1:
                    fig_pie = px.pie(
                        sentiment_counts, 
                        values='Count', 
                        names='Sentiment', 
                        title='Sentiment Category Distribution',
                        color='Sentiment',
                        color_discrete_map={'positive': '#10B981', 'negative': '#EF4444', 'neutral': '#6B7280'}
                    )
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col2:
                    fig_bar = px.bar(
                        sentiment_counts, 
                        x='Sentiment', 
                        y='Count', 
                        title='Sentiment Distribution Count',
                        color='Sentiment',
                        color_discrete_map={'positive': '#10B981', 'negative': '#EF4444', 'neutral': '#6B7280'}
                    )
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            st.success("Preprocessing completed! You can now proceed to the 'Model Training' page to fine-tune the model.")
