# FinSentiment – LLM-Based Financial Text Analyzer

FinSentiment is an end-to-end deep learning application that automatically classifies financial texts (news headlines, corporate announcements, earnings reports, and customer feedback) into **Positive**, **Negative**, or **Neutral** sentiment categories. It is powered by the domain-specific **FinBERT** transformer model, backed by an SQLite history database, and visualized via an interactive **Streamlit** dashboard.

---

## 🏗️ System Architecture & Workflow

### Architecture Diagram
```
┌────────────────────────────────────────────────────────────────────────┐
│                          Streamlit Dashboard (UI)                      │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Home / Info │  │ Dataset Upload│  │  Fine-Tuning │  │ Inference  │  │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│                           │                 │                │         │
└───────────────────────────┼─────────────────┼────────────────┼─────────┘
                            │                 │                │
                            ▼                 ▼                ▼
┌───────────────────────────┼─────────────────┼────────────────┼─────────┐
│                     Backend & Data Processing Core                   │
│                           │                 │                │
│  ┌────────────────────────▼──┐              │                │
│  │   Clean Text preprocess   │              │                │
│  └────────────────────────┬──┘              │                │
│                           │                 │                │
│  ┌────────────────────────▼──┐              │                │
│  │  Save Upload Metadata     ├──────────────┼────────┐       │
│  └───────────────────────────┘              │        │       │
│                                             ▼        │       ▼
│  ┌──────────────────────────────────────────┴──┐     │  ┌────┴─────┐
│  │      PyTorch fine-tuning (train.py)         │     │  │Predictor │
│  │   - Load FinBERT Pretrained (ProsusAI)       │     │  │ (XAI via │
│  │   - Fast mode (Freeze Backbone)             │     │  │Occlusion)│
│  │   - Compute Accuracy/Precision/Recall/F1     │     │  └────┬────┘
│  │   - Save fine-tuned checkpoint               │     │       │
│  └────────────────────────┬────────────────────┘     │       │
│                           │                          │       │
│                           ▼                          ▼       ▼
│  ┌────────────────────────┴──────────────────────────┴───────┴─────┐
│  │                      SQLite Database (finsentiment.db)          │
│  │   - predictions (manual/batch prediction inputs & scores)       │
│  │   - datasets (metadata tracking of CSV files)                   │
│  │   - model_metadata (classification run stats & configuration)   │
│  └─────────────────────────────────────────────────────────────────┘
└────────────────────────────────────────────────────────────────────────┘
```

### Inference & Explainable AI (XAI) Workflow
1. **Input text**: "Company reports record profits and increased revenue."
2. **Inference**: Tokenized sequence is sent to FinBERT; outputs a probability distribution.
3. **Occlusion-based XAI**:
   - The sentence is split into individual words.
   - For each word, we create a modified sentence with that word removed.
   - We run inference on the modified sentences and observe how much the confidence score of the predicted class drops.
   - Words causing the largest drops are marked as **Positive contributors** (green); words causing increases are **Negative contributors** (red).
4. **Display**: Under Prediction Page, a word heatmap highlights structural sentiment contributors.

---

## 📂 Project Structure

```
FinSentiment/
│
├── data/
│   ├── sample_financial_data.csv    # Automatically generated dummy news (100 rows)
│   └── temp_train_data.csv          # Temporary file created during active fine-tuning
├── models/
│   └── finbert_finetuned/           # Directory where fine-tuned local weights are saved
├── reports/
│   ├── loss_curves.png              # Loss charts generated post-training
│   ├── confusion_matrix.png         # Confusion matrix chart generated post-training
│   └── FinSentiment_Model_Report.pdf# Sample generated PDF performance report
├── database/
│   ├── db_manager.py                # Database queries and interface (SQLite)
│   └── finsentiment.db              # SQLite Database file (auto-generated)
├── app/
│   ├── dashboard.py                 # Streamlit App root & landing page
│   ├── utils.py                     # Custom CSS injections, Word Cloud, FPDF2 PDF helper
│   └── pages/
│       ├── 1_Dataset_Upload.py      # Previews and preprocesses CSV datasets
│       ├── 2_Model_Training.py      # Training controls, validation charts, PDF download
│       ├── 3_Prediction.py          # Manual prediction console, batch file predictions, XAI
│       ├── 4_Analytics.py           # Historical trends, gauge charts, news feed ticker
│       └── 5_About.py               # Scientific overview of the FinBERT model
├── training/
│   ├── preprocess.py                # Cleans HTML tags, URLs, and normalizes characters
│   └── train.py                     # Fine-tunes FinBERT using PyTorch
├── inference/
│   └── predictor.py                 # Predicts sentiment category and runs XAI logic
├── requirements.txt                 # Project python package definitions
├── README.md                        # Documentation
└── main.py                          # Launcher script
```

---

## ⚡ Installation & Local Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone the repository and navigate to the directory
```bash
git clone https://github.com/your-username/FinSentiment.git
cd FinSentiment
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```
*Note: Installs `transformers`, `torch` (PyTorch), `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `wordcloud`, and `fpdf2`.*

---

## 🚀 How to Run the Project

Launch the application using the master launcher file `main.py`. This script automatically initializes the SQLite database, checks directories, and starts the Streamlit development server.

```bash
python main.py
```
After running, Streamlit will print the local server address:
```
Local URL: http://localhost:8501
Network URL: http://192.168.1.10:8501
```
Open the URL in your web browser to interact with the dashboard.

---

## ⚙️ Model Fine-Tuning Modes
To run fine-tuning in Streamlit, navigate to the **Model Fine-Tuning & Evaluation** page. FinSentiment supports two training options:
1. **Fast Training Mode (Recommended for CPUs)**:
   - Freezes the BERT backbone parameters and trains only the sequence classification linear layer.
   - Drastically speeds up training (completes in ~1 minute on modern CPUs).
   - Prevents timeouts and avoids excessive RAM usage.
2. **Full Fine-Tuning**:
   - Updates all parameters across all transformer layers.
   - Recommended only on CUDA-enabled GPU servers.

---

## ☁️ Deployment Instructions for Streamlit Cloud

Streamlit Cloud is the easiest way to share this project. Follow these steps to deploy:

### 1. Structure Repository
Ensure your GitHub repository has the following files at the root level:
- `requirements.txt`
- `app/` folder (with all pages inside)
- `database/`, `training/`, `inference/` directories
- `main.py`

### 2. Configure Streamlit Cloud Settings
1. Log in to [Streamlit Share](https://share.streamlit.io/).
2. Click **New App** and select your repository, branch, and entrypoint file path:
   - **Main file path**: `app/dashboard.py` (Streamlit Cloud uses the main script to render, which will load sub-pages automatically).
3. Under **Advanced settings**, set your environment variable or select resources. Note that Streamlit Cloud provides standard CPU resources. We recommend using the pre-trained **FinBERT** models directly for inference and setting up training size constraints to stay within memory limits.

---

## 📄 Licensing & Credits
- Pre-trained model weights are provided by **ProsusAI** via the Hugging Face Model Hub.
- Visualizations are built using **Plotly Express** and **Matplotlib**.
- Reports are exported using **FPDF2**.
