"""
rag_patterns.py — RAG-based candlestick pattern matching using ChromaDB.

Pre-populates a ChromaDB collection with 15 classic candlestick patterns
and performs similarity search to find the top-2 matching patterns for
a given chart analysis.

Usage:
    from src.genai.rag_patterns import find_similar_patterns, initialize_pattern_db
    initialize_pattern_db()
    patterns = find_similar_patterns(prediction, confidence, rsi, macd)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import chromadb
from chromadb.config import Settings

from config import PATTERNS_DIR, CHROMA_COLLECTION


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLASSIC CANDLESTICK PATTERNS DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━
CANDLESTICK_PATTERNS = [
    {
        "id": "doji",
        "name": "Doji",
        "description": "A candle where open and close are virtually equal, forming a cross or plus sign. Indicates market indecision and potential trend reversal.",
        "signal": "Neutral/Reversal",
        "bullish_score": 0.0,
        "historical_accuracy": 0.52,
        "context": "Doji pattern shows indecision. Small body with shadows on both sides. Often appears at trend exhaustion points.",
    },
    {
        "id": "hammer",
        "name": "Hammer",
        "description": "A bullish reversal pattern with a small body at the top and a long lower shadow (at least 2x body). Appears after a downtrend.",
        "signal": "Bullish",
        "bullish_score": 0.75,
        "historical_accuracy": 0.60,
        "context": "Hammer pattern is bullish reversal. Small body at top, long lower wick. Buyers pushed price back up after sellers drove it down.",
    },
    {
        "id": "shooting_star",
        "name": "Shooting Star",
        "description": "A bearish reversal pattern with a small body at the bottom and a long upper shadow. Appears after an uptrend, suggesting rejection at higher levels.",
        "signal": "Bearish",
        "bullish_score": -0.70,
        "historical_accuracy": 0.59,
        "context": "Shooting star is bearish reversal. Small body at bottom, long upper wick. Sellers rejected the price at higher levels.",
    },
    {
        "id": "bullish_engulfing",
        "name": "Bullish Engulfing",
        "description": "A two-candle bullish reversal pattern where a large green candle completely engulfs the previous red candle. Strong bullish signal.",
        "signal": "Bullish",
        "bullish_score": 0.85,
        "historical_accuracy": 0.63,
        "context": "Bullish engulfing shows strong buying. Large green candle swallows previous red candle. Momentum shifting to buyers.",
    },
    {
        "id": "bearish_engulfing",
        "name": "Bearish Engulfing",
        "description": "A two-candle bearish reversal pattern where a large red candle completely engulfs the previous green candle. Strong bearish signal.",
        "signal": "Bearish",
        "bullish_score": -0.85,
        "historical_accuracy": 0.63,
        "context": "Bearish engulfing shows strong selling. Large red candle swallows previous green candle. Momentum shifting to sellers.",
    },
    {
        "id": "morning_star",
        "name": "Morning Star",
        "description": "A three-candle bullish reversal pattern: large red candle, small-body candle (gap down), then large green candle. Signals dawn of an uptrend.",
        "signal": "Bullish",
        "bullish_score": 0.80,
        "historical_accuracy": 0.65,
        "context": "Morning star is bullish reversal. Three candles: down, indecision, up. The star signals trend is about to reverse upward.",
    },
    {
        "id": "evening_star",
        "name": "Evening Star",
        "description": "A three-candle bearish reversal pattern: large green candle, small-body candle (gap up), then large red candle. Signals dusk of an uptrend.",
        "signal": "Bearish",
        "bullish_score": -0.80,
        "historical_accuracy": 0.65,
        "context": "Evening star is bearish reversal. Three candles: up, indecision, down. The star signals trend is about to reverse downward.",
    },
    {
        "id": "three_white_soldiers",
        "name": "Three White Soldiers",
        "description": "Three consecutive long green candles with progressively higher closes. Strong bullish continuation pattern indicating sustained buying pressure.",
        "signal": "Bullish",
        "bullish_score": 0.90,
        "historical_accuracy": 0.67,
        "context": "Three white soldiers is strong bullish. Three green candles in a row, each closing higher. Market has strong upward momentum.",
    },
    {
        "id": "three_black_crows",
        "name": "Three Black Crows",
        "description": "Three consecutive long red candles with progressively lower closes. Strong bearish continuation pattern indicating sustained selling pressure.",
        "signal": "Bearish",
        "bullish_score": -0.90,
        "historical_accuracy": 0.67,
        "context": "Three black crows is strong bearish. Three red candles in a row, each closing lower. Market has strong downward momentum.",
    },
    {
        "id": "harami",
        "name": "Harami",
        "description": "A two-candle pattern where a small candle is contained within the body of the preceding larger candle. Suggests potential reversal or consolidation.",
        "signal": "Neutral/Reversal",
        "bullish_score": 0.0,
        "historical_accuracy": 0.53,
        "context": "Harami pattern means 'pregnant'. Small candle inside previous large candle. Market is pausing, possible reversal ahead.",
    },
    {
        "id": "piercing_line",
        "name": "Piercing Line",
        "description": "A two-candle bullish reversal: red candle followed by green candle that opens below the low but closes above the midpoint of the red candle.",
        "signal": "Bullish",
        "bullish_score": 0.65,
        "historical_accuracy": 0.58,
        "context": "Piercing line is moderately bullish. Green candle opens low but closes above midpoint of previous red candle. Buyers stepping in.",
    },
    {
        "id": "dark_cloud_cover",
        "name": "Dark Cloud Cover",
        "description": "A two-candle bearish reversal: green candle followed by red candle that opens above the high but closes below the midpoint of the green candle.",
        "signal": "Bearish",
        "bullish_score": -0.65,
        "historical_accuracy": 0.58,
        "context": "Dark cloud cover is moderately bearish. Red candle opens high but closes below midpoint of previous green candle. Sellers taking over.",
    },
    {
        "id": "spinning_top",
        "name": "Spinning Top",
        "description": "A candle with small body and long shadows on both sides. Shows market indecision with neither buyers nor sellers in control.",
        "signal": "Neutral",
        "bullish_score": 0.0,
        "historical_accuracy": 0.50,
        "context": "Spinning top shows indecision. Small body, long wicks both sides. Market is undecided, waiting for a catalyst.",
    },
    {
        "id": "marubozu",
        "name": "Marubozu",
        "description": "A candle with no shadows (or very small ones) — all body. Green marubozu is very bullish, red marubozu is very bearish. Shows strong conviction.",
        "signal": "Strong Trend",
        "bullish_score": 0.50,
        "historical_accuracy": 0.62,
        "context": "Marubozu is a strong conviction candle. No wicks, all body. Shows complete dominance by either buyers or sellers.",
    },
    {
        "id": "tweezer",
        "name": "Tweezer Top/Bottom",
        "description": "Two candles with matching highs (top) or lows (bottom). Tweezers at the top suggest resistance, at the bottom suggest support.",
        "signal": "Reversal",
        "bullish_score": 0.0,
        "historical_accuracy": 0.56,
        "context": "Tweezer pattern shows price testing same level twice. At tops: double rejection, bearish. At bottoms: double support, bullish.",
    },
]


def get_chroma_client() -> chromadb.Client:
    """
    Get or create a ChromaDB persistent client.

    Returns:
        ChromaDB PersistentClient instance.
    """
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(PATTERNS_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def initialize_pattern_db(force_reset: bool = False) -> chromadb.Collection:
    """
    Initialize the ChromaDB collection with candlestick patterns.

    Populates the database with 15 classic patterns including their
    descriptions, signals, and historical accuracy data.

    Args:
        force_reset: If True, deletes and recreates the collection.

    Returns:
        The ChromaDB collection.
    """
    client = get_chroma_client()

    # Check if collection exists and has data
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
        if collection.count() >= len(CANDLESTICK_PATTERNS) and not force_reset:
            print(f"✅ Pattern database already initialized ({collection.count()} patterns)")
            return collection
        if force_reset:
            client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    # Create collection
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"description": "Classic candlestick patterns for RAG"},
    )

    # Add all patterns
    ids = []
    documents = []
    metadatas = []

    for pattern in CANDLESTICK_PATTERNS:
        ids.append(pattern["id"])
        documents.append(pattern["context"])
        metadatas.append({
            "name": pattern["name"],
            "description": pattern["description"],
            "signal": pattern["signal"],
            "bullish_score": pattern["bullish_score"],
            "historical_accuracy": pattern["historical_accuracy"],
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"✅ Initialized pattern database with {len(CANDLESTICK_PATTERNS)} patterns")
    return collection


def find_similar_patterns(
    prediction: str,
    confidence: float,
    rsi: float | None = None,
    macd: float | None = None,
    attention_description: str = "",
    top_k: int = 2,
) -> list[dict]:
    """
    Find the top-K most similar candlestick patterns using RAG.

    Constructs a query from the current analysis context and searches
    the ChromaDB collection for similar patterns.

    Args:
        prediction: Model prediction ('Up', 'Down', 'Neutral').
        confidence: Model confidence (0-1).
        rsi: RSI indicator value.
        macd: MACD indicator value.
        attention_description: Description of model attention regions.
        top_k: Number of similar patterns to return.

    Returns:
        List of pattern dictionaries with fields:
            - name: Pattern name
            - description: Pattern description
            - signal: Expected signal
            - bullish_score: Bullish/bearish score (-1 to 1)
            - historical_accuracy: Historical accuracy rate
            - similarity_score: Similarity to current chart
    """
    # Build query from current context
    trend_word = {
        "Up": "bullish upward buying",
        "Down": "bearish downward selling",
        "Neutral": "indecision sideways consolidation",
    }.get(prediction, "uncertain")

    rsi_context = ""
    if rsi is not None:
        if rsi > 70:
            rsi_context = " overbought conditions"
        elif rsi < 30:
            rsi_context = " oversold conditions"
        else:
            rsi_context = " moderate momentum"

    query = (
        f"Chart shows {trend_word} pattern with {confidence:.0%} confidence. "
        f"{attention_description} "
        f"Technical indicators suggest{rsi_context}."
    )

    try:
        client = get_chroma_client()
        collection = client.get_collection(CHROMA_COLLECTION)

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "documents", "distances"],
        )

        patterns = []
        if results and results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = max(0, 1.0 - distance / 2.0)  # Normalize distance to similarity

                patterns.append({
                    "name": metadata.get("name", "Unknown"),
                    "description": metadata.get("description", ""),
                    "signal": metadata.get("signal", "Unknown"),
                    "bullish_score": float(metadata.get("bullish_score", 0)),
                    "historical_accuracy": float(metadata.get("historical_accuracy", 0)),
                    "similarity_score": similarity,
                })

        return patterns

    except Exception as e:
        print(f"⚠️  RAG pattern search failed: {e}")
        # Return default patterns based on prediction
        return _get_default_patterns(prediction)


def _get_default_patterns(prediction: str) -> list[dict]:
    """
    Return default pattern matches when ChromaDB is unavailable.

    Args:
        prediction: Model prediction direction.

    Returns:
        List of 2 default pattern dictionaries.
    """
    defaults = {
        "Up": [
            {
                "name": "Bullish Engulfing",
                "description": "Large green candle engulfs previous red candle.",
                "signal": "Bullish",
                "bullish_score": 0.85,
                "historical_accuracy": 0.63,
                "similarity_score": 0.5,
            },
            {
                "name": "Hammer",
                "description": "Small body at top with long lower shadow.",
                "signal": "Bullish",
                "bullish_score": 0.75,
                "historical_accuracy": 0.60,
                "similarity_score": 0.4,
            },
        ],
        "Down": [
            {
                "name": "Bearish Engulfing",
                "description": "Large red candle engulfs previous green candle.",
                "signal": "Bearish",
                "bullish_score": -0.85,
                "historical_accuracy": 0.63,
                "similarity_score": 0.5,
            },
            {
                "name": "Shooting Star",
                "description": "Small body at bottom with long upper shadow.",
                "signal": "Bearish",
                "bullish_score": -0.70,
                "historical_accuracy": 0.59,
                "similarity_score": 0.4,
            },
        ],
        "Neutral": [
            {
                "name": "Doji",
                "description": "Open and close are virtually equal.",
                "signal": "Neutral/Reversal",
                "bullish_score": 0.0,
                "historical_accuracy": 0.52,
                "similarity_score": 0.5,
            },
            {
                "name": "Spinning Top",
                "description": "Small body with long shadows on both sides.",
                "signal": "Neutral",
                "bullish_score": 0.0,
                "historical_accuracy": 0.50,
                "similarity_score": 0.4,
            },
        ],
    }
    return defaults.get(prediction, defaults["Neutral"])


def get_pattern_signal_score(patterns: list[dict]) -> float:
    """
    Calculate a combined signal score from matched patterns.

    Returns a score between -1 (strongly bearish) and +1 (strongly bullish).

    Args:
        patterns: List of matched pattern dictionaries.

    Returns:
        Combined bullish/bearish score.
    """
    if not patterns:
        return 0.0

    total_weight = 0.0
    weighted_score = 0.0

    for pattern in patterns:
        weight = pattern.get("similarity_score", 0.5)
        score = pattern.get("bullish_score", 0.0)
        weighted_score += weight * score
        total_weight += weight

    if total_weight > 0:
        return weighted_score / total_weight
    return 0.0


if __name__ == "__main__":
    # Initialize and test
    print("🔧 Initializing pattern database...")
    initialize_pattern_db(force_reset=True)

    print("\n🔍 Testing pattern search...")
    patterns = find_similar_patterns(
        prediction="Up",
        confidence=0.78,
        rsi=55.0,
        macd=1.25,
        attention_description="Model focused on recent green candle bodies.",
    )

    print(f"\n📊 Found {len(patterns)} similar patterns:")
    for p in patterns:
        print(f"   • {p['name']} ({p['signal']})")
        print(f"     {p['description']}")
        print(f"     Similarity: {p['similarity_score']:.2f}, Historical accuracy: {p['historical_accuracy']:.0%}")

    score = get_pattern_signal_score(patterns)
    print(f"\n   Combined signal score: {score:.3f}")
