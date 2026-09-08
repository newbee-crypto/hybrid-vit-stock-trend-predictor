# Hybrid ViT Stock Trend Predictor

Hybrid deep learning project for short-term stock trend classification using candlestick chart images plus technical indicators such as `RSI` and `MACD`.

The model predicts `Down`, `Neutral`, or `Up`, and the app wraps the prediction in explainability, signal fusion, backtesting, and chat-based analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-ff4b4b)

## Why This Project

This project was built to move beyond pure chart-image classification.
- A Vision Transformer learns visual candlestick structure.
- Technical features add momentum and trend context.
- The result is shipped as an end-to-end dashboard instead of staying as a training notebook only.

## Key Result

- Best current test accuracy: `73%`
- Classes: `Down`, `Neutral`, `Up`
- Live app inputs: chart image + `RSI` + `MACD`

## Core Features

- Hybrid ViT classifier for 3-class trend prediction
- Volatility-aware labeling pipeline
- Grad-CAM and attention-based explainability
- Pattern retrieval and multi-signal fusion
- Streamlit app with analysis, chat, backtest, and performance pages

## Pipeline

1. Download OHLCV data with `src/data/collect_data.py`
2. Generate 224x224 candlestick images with `src/data/generate_charts.py`
3. Build labels and indicators with `src/data/label_data.py`
4. Train from `notebooks/train-2-vit.ipynb`
5. Load `models/checkpoints/best_model.pth`
6. Run the dashboard from `app/app.py`

## Run Locally

```bash
git clone <your-repo-url>
cd hybrid-vit-stock-trend-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Optional env vars:

```bash
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Without API keys, the core prediction flow still works, but LLM and news features may be limited.

## Repo Structure

```text
src/data/         data collection, chart generation, labeling
src/model/        hybrid ViT training, loading, inference, evaluation
src/xai/          Grad-CAM and attention visualization
src/genai/        explanations, RAG patterns, signal fusion, analyst agent
src/backtesting/  backtest logic
app/              Streamlit dashboard
notebooks/        training notebook
```

## Notes

- Built for educational and portfolio use
- Not financial advice
- Currently kept private while being polished
