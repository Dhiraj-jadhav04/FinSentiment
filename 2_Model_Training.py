import streamlit as st
import os
import pandas as pd
from training.train import train_model
from app.utils import inject_custom_css, generate_pdf_report
import database.db_manager as db

inject_custom_css()

st.markdown('<div class="gradient-header">Model Fine-Tuning & Evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subheader">Configure parameters and train the FinBERT model on custom financial data</div>', unsafe_allow_html=True)

# Check if dataset is in session state
if 'cleaned_df' not in st.session_state:
    st.warning("Please upload and preprocess a dataset in the 'Dataset Upload' page first.")
else:
    df = st.session_state['cleaned_df']
    filename = st.session_state['dataset_filename']
    
    st.info(f"Active Dataset: **{filename}** ({len(df)} rows ready for training)")
    
    # Parameters Columns
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### Hyperparameter Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.slider("Epochs", min_value=1, max_value=10, value=3)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        
    with col2:
        lr = st.select_slider("Learning Rate", options=[1e-5, 2e-5, 3e-5, 5e-5], value=2e-5)
        fast_mode = st.checkbox("Fast Training Mode (Classifier Head Only)", value=True, 
                                help="Freeze BERT transformer backbone layers and train only the output head. Recommended for CPU to complete training in ~1-2 minutes.")
        
    with col3:
        st.markdown("""
        **Training Guidelines:**
        - **Fast Mode** freezes BERT weights. Updates are 10x faster.
        - **Learning Rate**: 2e-5 is recommended for sequence classification.
        - Training saves checkpoints to the local `models/` directory.
        """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Start FinBERT Fine-Tuning"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Save active dataset to a temporary CSV for train script
        temp_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "temp_train_data.csv")
        df.to_csv(temp_csv_path, index=False)
        
        # Progress callback
        def progress_update(prog, text):
            progress_bar.progress(prog)
            status_text.text(text)
            
        try:
            with st.spinner("Fine-tuning FinBERT... (this will take a moment)"):
                results = train_model(
                    data_path=temp_csv_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    fast_mode=fast_mode,
                    progress_callback=progress_update
                )
                
            st.success("Training completed successfully!")
            
            # Render evaluation results
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("### Model Evaluation Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
            m_col2.metric("Precision", f"{results['precision']*100:.2f}%")
            m_col3.metric("Recall", f"{results['recall']*100:.2f}%")
            m_col4.metric("F1-Score", f"{results['f1']*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show images
            col_img1, col_img2 = st.columns(2)
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
            curves_path = os.path.join(reports_dir, "loss_curves.png")
            cm_path = os.path.join(reports_dir, "confusion_matrix.png")
            
            with col_img1:
                if os.path.exists(curves_path):
                    st.image(curves_path, caption="Training Loss Curves", use_container_width=True)
            with col_img2:
                if os.path.exists(cm_path):
                    st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                    
            # PDF Generation
            st.markdown("### Export Performance Report")
            metrics = {
                "accuracy": results['accuracy'],
                "precision": results['precision'],
                "recall": results['recall'],
                "f1": results['f1']
            }
            
            pdf_bytes = generate_pdf_report(metrics, curves_path, cm_path)
            st.download_button(
                label="📥 Download Performance PDF Report",
                data=pdf_bytes,
                file_name="FinSentiment_Model_Report.pdf",
                mime="application/pdf"
            )
            
            # Cleanup temp CSV
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                
        except Exception as e:
            st.error(f"Error occurred during training: {e}")

# Display latest saved model metadata
st.markdown("---")
st.markdown("### Model Train History Registry")
latest_meta = db.get_latest_model_meta()
if latest_meta:
    st.json(latest_meta)
else:
    st.info("No training runs registered in the database database yet.")
