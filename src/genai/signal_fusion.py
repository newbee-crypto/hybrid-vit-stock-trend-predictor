"""
signal_fusion.py — Weighted signal fusion combining multiple analysis sources.

Combines ViT prediction, RAG pattern matching, news sentiment, and
technical indicators into a single actionable signal with confidence.

Usage:
    from src.genai.signal_fusion import fuse_signals
    result = fuse_signals(vit_result, rag_patterns, ticker)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from dotenv import load_dotenv

from config import FUSION_WEIGHTS

load_dotenv()


def get_vit_signal(
    prediction: str,
    confidence: float,
    probabilities: dict[str, float],
) -> dict:
    """
    Convert ViT prediction to a normalized signal.

    Args:
        prediction: Predicted class ('Up', 'Down', 'Neutral').
        confidence: Model confidence (0-1).
        probabilities: Dict of class probabilities.

    Returns:
        Signal dict with score (-1 to +1), signal name, and confidence.
    """
    # Convert prediction to directional score
    up_prob = probabilities.get("Up", 0)
    down_prob = probabilities.get("Down", 0)

    # Score = up_prob - down_prob, weighted by confidence
    score = (up_prob - down_prob) * confidence

    signal = "Buy" if score > 0.1 else "Sell" if score < -0.1 else "Hold"

    return {
        "score": score,
        "signal": signal,
        "confidence": confidence,
        "prediction": prediction,
    }


def get_rag_signal(patterns: list[dict]) -> dict:
    """
    Convert RAG pattern matches to a signal.

    Args:
        patterns: List of matched pattern dicts from rag_patterns.py.

    Returns:
        Signal dict with score, signal name, and pattern info.
    """
    if not patterns:
        return {"score": 0.0, "signal": "Hold", "confidence": 0.0, "patterns": []}

    # Weighted average of pattern scores
    total_weight = 0.0
    weighted_score = 0.0

    for pattern in patterns:
        sim = pattern.get("similarity_score", 0.5)
        bullish = pattern.get("bullish_score", 0.0)
        accuracy = pattern.get("historical_accuracy", 0.5)
        weight = sim * accuracy
        weighted_score += weight * bullish
        total_weight += weight

    score = weighted_score / total_weight if total_weight > 0 else 0.0
    confidence = min(1.0, total_weight / len(patterns))

    signal = "Buy" if score > 0.1 else "Sell" if score < -0.1 else "Hold"

    return {
        "score": score,
        "signal": signal,
        "confidence": confidence,
        "patterns": [p.get("name", "Unknown") for p in patterns],
    }


def get_news_signal(ticker: str) -> dict:
    """
    Get news sentiment signal using Tavily API.

    Searches for recent news about the stock and analyzes sentiment.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        Signal dict with score, signal name, and news summary.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {
                "score": 0.0,
                "signal": "Hold",
                "confidence": 0.0,
                "summary": "News API not configured",
            }

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        # Search for recent stock news
        response = client.search(
            query=f"{ticker} stock price prediction forecast today",
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        # Analyze sentiment from search results
        answer = response.get("answer", "")
        results = response.get("results", [])

        # Simple keyword-based sentiment scoring
        bullish_keywords = [
            "bullish", "up", "gain", "rise", "rally", "positive",
            "growth", "buy", "upgrade", "outperform", "strong",
            "beat", "exceed", "higher", "surge", "momentum",
        ]
        bearish_keywords = [
            "bearish", "down", "loss", "fall", "decline", "negative",
            "sell", "downgrade", "underperform", "weak", "miss",
            "below", "lower", "crash", "drop", "risk",
        ]

        # Combine all text
        all_text = answer.lower()
        for r in results:
            all_text += " " + r.get("content", "").lower()

        # Count sentiment keywords
        bullish_count = sum(1 for word in bullish_keywords if word in all_text)
        bearish_count = sum(1 for word in bearish_keywords if word in all_text)
        total = bullish_count + bearish_count

        if total > 0:
            score = (bullish_count - bearish_count) / total
        else:
            score = 0.0

        # Clamp score to [-1, 1]
        score = max(-1.0, min(1.0, score))

        confidence = min(1.0, total / 10.0)  # Higher keyword count = higher confidence
        signal = "Buy" if score > 0.1 else "Sell" if score < -0.1 else "Hold"

        # Create summary from top results
        summary = answer[:200] if answer else "No news summary available."

        return {
            "score": score,
            "signal": signal,
            "confidence": confidence,
            "summary": summary,
            "num_articles": len(results),
        }

    except Exception as e:
        return {
            "score": 0.0,
            "signal": "Hold",
            "confidence": 0.0,
            "summary": f"News analysis error: {str(e)[:100]}",
        }


def get_technical_signal(
    rsi: float | None = None,
    macd: float | None = None,
    macd_signal_val: float | None = None,
) -> dict:
    """
    Convert technical indicators to a signal.

    Combines RSI and MACD into a single directional score.

    Args:
        rsi: RSI value (0-100).
        macd: MACD line value.
        macd_signal_val: MACD signal line value.

    Returns:
        Signal dict with score, signal name, and indicator values.
    """
    scores = []

    # RSI signal
    if rsi is not None:
        if rsi > 70:
            rsi_score = -0.5 * ((rsi - 70) / 30)  # Overbought → bearish
        elif rsi < 30:
            rsi_score = 0.5 * ((30 - rsi) / 30)  # Oversold → bullish
        else:
            rsi_score = (rsi - 50) / 50 * 0.3  # Neutral zone: slight directional bias
        scores.append(("RSI", rsi_score))

    # MACD signal
    if macd is not None and macd_signal_val is not None:
        macd_diff = macd - macd_signal_val
        # Normalize MACD difference (rough heuristic)
        macd_score = max(-1.0, min(1.0, macd_diff * 10))
        scores.append(("MACD", macd_score * 0.5))

    if not scores:
        return {
            "score": 0.0,
            "signal": "Hold",
            "confidence": 0.0,
            "rsi": rsi,
            "macd": macd,
        }

    # Average the indicator scores
    total_score = sum(s for _, s in scores) / len(scores)
    total_score = max(-1.0, min(1.0, total_score))

    confidence = 0.5 + abs(total_score) * 0.5  # More extreme = more confident
    signal = "Buy" if total_score > 0.1 else "Sell" if total_score < -0.1 else "Hold"

    return {
        "score": total_score,
        "signal": signal,
        "confidence": confidence,
        "rsi": rsi,
        "macd": macd,
        "details": {name: score for name, score in scores},
    }


def fuse_signals(
    vit_result: dict | None = None,
    rag_patterns: list[dict] | None = None,
    ticker: str = "AAPL",
    rsi: float | None = None,
    macd: float | None = None,
    macd_signal_val: float | None = None,
    weights: dict | None = None,
) -> dict:
    """
    Combine all signal sources into a single actionable signal.

    Signal Fusion Formula:
        Combined Score = w_vit * ViT + w_rag * RAG + w_news * News + w_tech * Tech

    Args:
        vit_result: ViT prediction result dict.
        rag_patterns: RAG matched patterns list.
        ticker: Stock ticker for news search.
        rsi: RSI indicator value.
        macd: MACD line value.
        macd_signal_val: MACD signal line value.
        weights: Custom fusion weights. Defaults to config weights.

    Returns:
        Dictionary with:
            - signal: 'Buy', 'Sell', or 'Hold'
            - confidence: Combined confidence (0-1)
            - combined_score: Weighted score (-1 to +1)
            - components: Individual signal breakdowns
    """
    weights = weights or FUSION_WEIGHTS

    # Get individual signals
    vit_sig = get_vit_signal(
        prediction=vit_result.get("prediction", "Neutral") if vit_result else "Neutral",
        confidence=vit_result.get("confidence", 0.5) if vit_result else 0.5,
        probabilities=vit_result.get("probabilities", {"Down": 0.33, "Neutral": 0.34, "Up": 0.33}) if vit_result else {"Down": 0.33, "Neutral": 0.34, "Up": 0.33},
    )

    rag_sig = get_rag_signal(rag_patterns or [])
    news_sig = get_news_signal(ticker)
    tech_sig = get_technical_signal(rsi, macd, macd_signal_val)

    # Weighted combination
    combined_score = (
        weights["vit"] * vit_sig["score"]
        + weights["rag"] * rag_sig["score"]
        + weights["news"] * news_sig["score"]
        + weights["technical"] * tech_sig["score"]
    )

    # Combined confidence (weighted average of individual confidences)
    combined_confidence = (
        weights["vit"] * vit_sig["confidence"]
        + weights["rag"] * rag_sig["confidence"]
        + weights["news"] * news_sig["confidence"]
        + weights["technical"] * tech_sig["confidence"]
    )

    # Determine final signal
    if combined_score > 0.1:
        final_signal = "Buy"
    elif combined_score < -0.1:
        final_signal = "Sell"
    else:
        final_signal = "Hold"

    # Agreement analysis
    signals = [vit_sig["signal"], rag_sig["signal"], news_sig["signal"], tech_sig["signal"]]
    agreement = signals.count(final_signal) / len(signals)

    return {
        "signal": final_signal,
        "confidence": min(1.0, combined_confidence),
        "combined_score": max(-1.0, min(1.0, combined_score)),
        "agreement": agreement,
        "components": {
            "vit": vit_sig,
            "rag": rag_sig,
            "news": news_sig,
            "technical": tech_sig,
        },
        "weights": weights,
        "ticker": ticker,
    }


if __name__ == "__main__":
    # Quick test
    result = fuse_signals(
        vit_result={
            "prediction": "Up",
            "confidence": 0.78,
            "probabilities": {"Down": 0.10, "Neutral": 0.12, "Up": 0.78},
        },
        rag_patterns=[
            {"name": "Bullish Engulfing", "bullish_score": 0.85, "similarity_score": 0.7, "historical_accuracy": 0.63},
        ],
        ticker="AAPL",
        rsi=55.0,
        macd=1.25,
        macd_signal_val=0.98,
    )

    print(f"\n{'='*60}")
    print(f"📊 Signal Fusion Result:")
    print(f"   Final Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Combined Score: {result['combined_score']:+.3f}")
    print(f"   Agreement: {result['agreement']:.0%}")
    print(f"\n   Component Breakdown:")
    for name, sig in result["components"].items():
        weight = result["weights"][name]
        print(f"     {name:>10} ({weight:.0%}): {sig['signal']:>4} (score: {sig['score']:+.3f})")
    print(f"{'='*60}")
