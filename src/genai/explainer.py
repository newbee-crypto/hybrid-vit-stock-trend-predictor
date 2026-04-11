"""
explainer.py — LLM-powered explanation generator using Google Gemini.

Takes prediction results, attention analysis, and technical indicators
to generate plain-English explanations a retail investor can understand.

Usage:
    from src.genai.explainer import explain_prediction
    explanation = explain_prediction(prediction_result, attention_result, technicals)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from dotenv import load_dotenv
import google.generativeai as genai

from config import GEMINI_MODEL

load_dotenv()


def _get_gemini_model() -> genai.GenerativeModel:
    """
    Initialize and return the Gemini generative model.

    Returns:
        Configured GenerativeModel instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Set it in your .env file or environment variables."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def explain_prediction(
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
    attention_description: str = "",
    rsi: float | None = None,
    macd: float | None = None,
    trend_score: float | None = None,
    ticker: str = "the stock",
    similar_patterns: list[dict] | None = None,
) -> str:
    """
    Generate a plain-English explanation of the model's prediction.

    Uses Gemini to create a 3-4 sentence explanation that a retail
    investor can easily understand.

    Args:
        prediction: Predicted trend ('Up', 'Down', 'Neutral').
        confidence: Model confidence (0-1).
        probabilities: Dict mapping class names to probabilities.
        attention_description: Natural language description of attention.
        rsi: RSI indicator value (0-100).
        macd: MACD line value.
        trend_score: Volatility-adjusted trend score used by the classifier.
        ticker: Stock ticker symbol.
        similar_patterns: List of similar candlestick patterns from RAG.

    Returns:
        Human-readable explanation string.
    """
    # Build context for the prompt
    prob_text = ", ".join(
        f"{name}: {prob:.1%}" for name, prob in probabilities.items()
    )

    technical_text = ""
    if rsi is not None:
        rsi_condition = (
            "overbought" if rsi > 70
            else "oversold" if rsi < 30
            else "neutral"
        )
        technical_text += f"RSI is {rsi:.1f} ({rsi_condition}). "
    if macd is not None:
        technical_text += f"MACD is {macd:.4f}. "
    if trend_score is not None:
        trend_bias = "bullish" if trend_score > 0.5 else "bearish" if trend_score < -0.5 else "neutral"
        technical_text += f"Trend score is {trend_score:.2f}, which suggests a {trend_bias} setup."

    pattern_text = ""
    if similar_patterns:
        pattern_names = [p.get("name", "Unknown") for p in similar_patterns[:2]]
        pattern_text = f"Similar candlestick patterns detected: {', '.join(pattern_names)}."

    prompt = f"""You are a professional stock market analyst explaining a chart analysis to a retail investor.

ANALYSIS RESULTS:
- Stock: {ticker}
- AI Model Prediction: {prediction} trend
- Confidence: {confidence:.1%}
- Class probabilities: {prob_text}
- Model attention: {attention_description if attention_description else 'Not available'}
- Technical indicators: {technical_text if technical_text else 'Not available'}
- {pattern_text if pattern_text else ''}

INSTRUCTIONS:
1. Write exactly 3-4 sentences explaining the prediction in SIMPLE English
2. Mention the prediction and confidence level
3. Reference the chart patterns or technical indicators that support (or contradict) the prediction
4. DO NOT use jargon without explaining it
5. Include a brief risk disclaimer
6. Be conversational and helpful, not robotic

EXAMPLE OUTPUT:
"Based on the candlestick chart analysis, our AI model predicts an upward trend for AAPL with 78% confidence. The recent chart shows a series of higher lows with increasing volume, which typically signals growing buyer interest. The RSI at 55 supports this view, sitting in a healthy middle range without being overbought. However, always remember that past patterns don't guarantee future results — consider this as one input alongside your own research."

Now write the explanation:"""

    try:
        model = _get_gemini_model()
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=300,
                top_p=0.9,
            ),
        )
        explanation = response.text.strip()

        # Clean up quotes if present
        if explanation.startswith('"') and explanation.endswith('"'):
            explanation = explanation[1:-1]

        return explanation

    except Exception as e:
        # Fallback explanation if Gemini fails
        return _fallback_explanation(prediction, confidence, ticker, rsi, macd)


def _fallback_explanation(
    prediction: str,
    confidence: float,
    ticker: str,
    rsi: float | None = None,
    macd: float | None = None,
) -> str:
    """
    Generate a fallback explanation when the Gemini API is unavailable.

    Args:
        prediction: Predicted trend direction.
        confidence: Model confidence.
        ticker: Stock ticker symbol.
        rsi: RSI value.
        macd: MACD value.

    Returns:
        Basic explanation string.
    """
    direction = {
        "Up": "an upward",
        "Down": "a downward",
        "Neutral": "a sideways/neutral",
    }.get(prediction, "an uncertain")

    explanation = (
        f"The AI model predicts {direction} trend for {ticker} "
        f"with {confidence:.0%} confidence based on the candlestick chart pattern analysis."
    )

    if rsi is not None:
        if rsi > 70:
            explanation += f" The RSI at {rsi:.0f} suggests the stock may be overbought."
        elif rsi < 30:
            explanation += f" The RSI at {rsi:.0f} suggests the stock may be oversold."
        else:
            explanation += f" The RSI at {rsi:.0f} indicates moderate momentum."

    if macd is not None:
        explanation += f" MACD analysis provides additional confirmation of this trend."

    explanation += (
        " Please note: This is an AI-generated analysis for educational purposes only. "
        "Always do your own research before making investment decisions."
    )

    return explanation


def explain_signal_fusion(
    fusion_result: dict,
    ticker: str = "the stock",
) -> str:
    """
    Generate an explanation of the signal fusion (combined signals) result.

    Args:
        fusion_result: Dictionary from signal_fusion.py with combined signal.
        ticker: Stock ticker symbol.

    Returns:
        Human-readable explanation of the combined signal.
    """
    prompt = f"""You are a stock analyst summarizing a multi-factor analysis for a retail investor.

COMBINED SIGNAL ANALYSIS for {ticker}:
- Final Signal: {fusion_result.get('signal', 'N/A')}
- Combined Confidence: {fusion_result.get('confidence', 0):.1%}

Component Breakdown:
1. AI Chart Analysis (40% weight): {fusion_result.get('components', {}).get('vit', {}).get('signal', 'N/A')} ({fusion_result.get('components', {}).get('vit', {}).get('confidence', 0):.1%} confident)
2. Historical Pattern Match (20% weight): {fusion_result.get('components', {}).get('rag', {}).get('signal', 'N/A')}
3. News Sentiment (25% weight): {fusion_result.get('components', {}).get('news', {}).get('signal', 'N/A')} ({fusion_result.get('components', {}).get('news', {}).get('score', 0):.2f})
4. Technical Indicators (15% weight): {fusion_result.get('components', {}).get('technical', {}).get('signal', 'N/A')}

Write 2-3 sentences summarizing this multi-factor analysis. Be clear about whether the signals agree or conflict. Keep it simple."""

    try:
        model = _get_gemini_model()
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=200,
            ),
        )
        return response.text.strip()
    except Exception:
        return (
            f"The combined analysis for {ticker} shows a "
            f"{fusion_result.get('signal', 'neutral')} signal with "
            f"{fusion_result.get('confidence', 0):.0%} confidence, "
            f"based on chart patterns, news sentiment, and technical indicators."
        )


if __name__ == "__main__":
    # Quick test
    result = explain_prediction(
        prediction="Up",
        confidence=0.78,
        probabilities={"Down": 0.10, "Neutral": 0.12, "Up": 0.78},
        attention_description="The model focused on recent candle bodies showing consecutive green candles.",
        rsi=55.3,
        macd=1.25,
        trend_score=0.84,
        ticker="AAPL",
    )
    print(result)
