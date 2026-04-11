# Hybrid ViT Stock Trend Predictor

> AI-powered stock trend prediction from candlestick charts plus RSI, MACD, and trend score, with explainability, signal fusion, and a Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Hybrid ViT Prediction** | Vision Transformer trained on chart images plus RSI, MACD, and trend score |
| 🔥 **Grad-CAM** | Heatmap visualization showing where the model focuses |
| 🧠 **AI Explanation** | Gemini-powered plain-English analysis |
| 🔄 **RAG Patterns** | ChromaDB matching against 15 classic candlestick patterns |
| ⚡ **Signal Fusion** | Combined signal: ViT(40%) + RAG(20%) + News(25%) + Tech(15%) |
| 💬 **AI Chat** | LangChain analyst agent with memory |
| 📈 **Backtesting** | Strategy testing with equity curves and Sharpe ratio |
| 🎯 **LLM Judge** | Automated explanation quality scoring |

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd hybrid-vit-stock-trend-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the example env file
cp .env.example .env

# Edit .env with your actual keys:
# GOOGLE_API_KEY=your_gemini_api_key
# TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run Data Pipeline (Local)

```bash
# Step 1: Download stock data
python src/data/collect_data.py

# Step 2: Generate candlestick chart images
python src/data/generate_charts.py

# Step 3: Create labeled dataset with indicators
python src/data/label_data.py
```

### 4. Train Model (Google Colab)

1. Upload `data/charts/` and `data/labels/labeled_dataset.csv` to Google Drive
2. Open `notebooks/train-2-vit.ipynb` in Kaggle or Colab
3. Select GPU runtime (T4 or A100)
4. Run all cells
5. Download the checkpoint and place it at: `models/checkpoints/best_model.pth`

> **💡 Tip:** The app works in **fallback mode** without a trained model — predictions will use the pretrained ViT backbone.

### 5. Launch the App

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
├── config.py                      # Global configuration
├── requirements.txt               # Dependencies
├── .env.example                   # API key template
├── data/
│   ├── raw/                       # OHLCV CSVs
│   ├── charts/                    # Candlestick images (224×224)
│   ├── labels/                    # Labeled datasets
│   └── patterns/                  # ChromaDB store
├── models/checkpoints/            # Trained model weights
├── notebooks/
│   └── train-2-vit.ipynb          # Latest hybrid-model training notebook
├── src/
│   ├── data/                      # Data pipeline
│   ├── model/                     # ViT model + training
│   ├── xai/                       # Grad-CAM + attention
│   ├── genai/                     # Gemini + RAG + fusion + agent
│   ├── memory/                    # Chat memory
│   ├── backtesting/               # Strategy backtester
│   └── evaluation/                # LLM judge
├── app/
│   ├── app.py                     # Streamlit main
│   ├── pages/                     # Multi-page app
│   └── utils/                     # UI helpers
├── Dockerfile                     # Docker deployment
├── render.yaml                    # Render.com config
├── ARCHITECTURE.md                # System architecture
└── DEVLOG.md                      # Development log
```

---

## 📊 Pages

### 1. Analyze Chart
- Upload a chart image OR select stock + date
- View: prediction, confidence, Grad-CAM heatmap, AI explanation
- See similar patterns from RAG and combined signal fusion

### 2. Chat
- Talk with the AI analyst about any stock
- Memory: remembers last 5 turns and your interests
- Suggested prompts for quick access

### 3. Backtest
- Select stock and initial capital
- Run the prediction strategy on historical data
- View: return, win rate, Sharpe ratio, equity curve

### 4. Performance
- Model accuracy, F1 scores, confusion matrix
- Training loss/accuracy curves
- LLM judge quality scores (accuracy, clarity, alignment)

---

## ⚡ Signal Fusion

The system combines 4 signal sources:

```
Combined = 0.40 × ViT + 0.20 × RAG + 0.25 × News + 0.15 × Technical
```

| Component | Weight | Source |
|-----------|--------|--------|
| ViT Model | 40% | Candlestick chart image prediction |
| RAG Patterns | 20% | ChromaDB similarity to classic patterns |
| News Sentiment | 25% | Tavily API keyword sentiment |
| Technical | 15% | RSI (14) + MACD-based momentum |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t candlestick-vit .

# Run
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  candlestick-vit
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. The predictions, signals, and analyses provided by this system should **NOT** be used as the sole basis for real investment decisions. Always do your own research and consult with a qualified financial advisor.

---

## Privacy

This repo is currently intended to stay private while the project is being polished for portfolio use.
