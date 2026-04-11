# DEVLOG — Candlestick ViT Stock Trend Predictor

## 2026-04-04 — Project Initialization (Agent 1-11)

### Agent 1 — Setup ✅
- Created project folder structure with all required directories
- `config.py` — Centralized configuration: 8 stock tickers, window=20, ViT config, fusion weights
- `requirements.txt` — All dependencies pinned (torch, timm, langchain, streamlit, etc.)
- `.env.example` — API key template (GOOGLE_API_KEY, TAVILY_API_KEY)
- `.gitignore` — Python/ML/IDE ignore patterns

### Agent 2 — Data Pipeline ✅
- `collect_data.py` — Downloads 3 years of OHLCV data per ticker via yfinance
  - Validates data integrity (High >= Low, Volume >= 0)
  - Handles NaN cleanup and date formatting
- `generate_charts.py` — Creates 224×224 candlestick images
  - Custom dark style (green up, red down, black background)
  - Sliding window approach with no axes/labels (clean charts for model)
  - Saves metadata CSV alongside charts
- `label_data.py` — Labels with trend direction
  - Forward return over 5 days: Up(>+2%), Down(<-2%), Neutral
  - Computes RSI(14) and MACD(12,26,9) via pandas-ta
  - Time-based split: 70/15/15 train/val/test (no lookahead bias)

### Agent 3 — ViT Model ✅
- `vit_model.py` — CandlestickViT class
  - Backbone: vit_base_patch16_224 (timm, pretrained)
  - Custom head: LayerNorm → Dropout → Linear(768,256) → GELU → Dropout → Linear(256,3)
  - Attention extraction method for XAI
- `train.py` — Full training pipeline
  - Data augmentation: RandomHorizontalFlip, ColorJitter, RandomRotation
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Early stopping (patience=5), best checkpoint saving
- `evaluate.py` — Test set evaluation
  - Per-class precision/recall/F1, macro F1, confusion matrix heatmap
- `inference.py` — Single-image and batch prediction
  - Fallback mode for untrained models
- `train_colab.ipynb` — Google Colab notebook
  - Drive mount, GPU check, full training loop, visualization, download

### Agent 4 — XAI ✅
- `gradcam.py` — Grad-CAM for Vision Transformer
  - Hooks on last transformer block's norm layer
  - 14×14 spatial heatmap from patch-level gradients
  - Overlay generation with matplotlib jet colormap
- `attention_viz.py` — Attention analysis
  - CLS token attention to all patches
  - 9-region grid mapping (top/mid/bottom × left/center/right)
  - Natural language description of focus areas with interpretation

### Agent 5 — GenAI & RAG ✅
- `explainer.py` — Gemini explanation generator
  - Structured prompt with context (prediction, confidence, RSI, MACD, attention)
  - Few-shot example in system prompt
  - Fallback explanation when API unavailable
- `rag_patterns.py` — ChromaDB pattern database
  - 15 classic patterns: Doji, Hammer, Engulfing, Morning Star, etc.
  - Each pattern has: description, signal, bullish_score, historical_accuracy
  - Similarity search returns top-2 matching patterns
- `signal_fusion.py` — Multi-signal combination
  - ViT(40%) + RAG(20%) + News(25%) + Technical(15%)
  - News via Tavily API with keyword-based sentiment scoring
  - Technical signals from RSI/MACD interpretation
  - Agreement analysis across all components
- `analyst_agent.py` — LangChain conversational agent
  - 5 tools: analyze_chart, get_news, find_patterns, get_fusion_signal, get_technicals
  - Gemini LLM backbone with professional analyst system prompt
  - Supports conversation history for context continuity

### Agent 6 — Memory ✅
- `chat_memory.py` — ConversationBufferWindowMemory-style implementation
  - Sliding window of last 5 turns
  - Stock ticker frequency tracking (regex extraction)
  - Interest detection: 3+ queries on same stock triggers personalization
  - Personalized messages: "Since you've been watching TSLA..."
  - Session statistics tracking

### Agent 7 — Backtesting ✅
- `backtest.py` — Strategy backtester
  - Strategy: Up→Buy, Down→Sell, Neutral→Hold
  - Metrics: total return, win rate, Sharpe ratio, max drawdown
  - Plotly interactive equity curve with buy/sell markers
  - JSON report + matplotlib static plot
  - Configurable initial capital and commission

### Agent 8 — Evaluation ✅
- `llm_judge.py` — LLM-based explanation scoring
  - Three dimensions: Accuracy(1-5), Clarity(1-5), Alignment(1-5)
  - Gemini as judge with structured JSON output
  - Batch evaluation with aggregate scoring
  - Fallback scores when API unavailable

### Agent 9 — Frontend (Streamlit) ✅
- `app.py` — Main entry point
  - Premium dark theme with custom CSS
  - Inter font, glassmorphism cards, gradient accents
  - Session state for memory, agent, model
- `1_Analyze_Chart.py` — Full analysis page
  - Upload image OR select stock + date
  - Prediction banner → Grad-CAM → Attention → Explanation → Patterns → Signal Fusion
  - All components with error handling and graceful degradation
- `2_Chat.py` — Agent chat interface
  - Suggested prompts for new users
  - Session info sidebar (turns, stocks queried, interests)
  - Memory-powered personalization
- `3_Backtest.py` — Backtesting dashboard
  - Stock selector, capital input, run button
  - 8 metric cards + interactive equity curve + trade log
- `4_Performance.py` — Model metrics dashboard
  - Confusion matrix (Plotly heatmap), per-class expandable metrics
  - Training loss/accuracy/LR curves
  - LLM judge radar chart
- `ui_helpers.py` — Custom CSS and HTML components
  - metric_card, signal_badge, prediction_banner, pattern_card
  - confidence_bar, section_header, custom_divider

### Agent 10 — DevOps ✅
- `Dockerfile` — Python 3.11 slim, Streamlit server on port 8501
- `render.yaml` — Render.com web service config with env vars
- `.github/workflows/ci.yml` — GitHub Actions CI
  - Jobs: lint (flake8), test (config + imports), docker build

### Agent 11 — Documentation ✅
- `ARCHITECTURE.md` — Full system architecture with Mermaid diagrams
- `DEVLOG.md` — This file, documenting every agent's work
- `README.md` — User-facing documentation with setup guide
