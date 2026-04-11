"""
llm_judge.py — LLM-based evaluation of model explanations.

Uses Gemini to score the quality of AI-generated stock analysis
explanations across three dimensions: accuracy, clarity, and alignment.

Usage:
    from src.evaluation.llm_judge import evaluate_explanation
    scores = evaluate_explanation(prediction, explanation, attention_desc)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

from config import GEMINI_MODEL, CHECKPOINTS_DIR

load_dotenv()


def evaluate_explanation(
    prediction: str,
    confidence: float,
    explanation: str,
    attention_description: str = "",
    rsi: float | None = None,
    macd: float | None = None,
    ticker: str = "STOCK",
) -> dict:
    """
    Use Gemini as an LLM judge to score an explanation.

    Evaluates across three dimensions:
    1. Accuracy (1-5): Does the explanation match the prediction?
    2. Clarity (1-5): Is it understandable by a non-expert?
    3. Alignment (1-5): Does it reference the right chart features?

    Args:
        prediction: The model's trend prediction ('Up', 'Down', 'Neutral').
        confidence: Prediction confidence (0-1).
        explanation: The AI-generated explanation text.
        attention_description: Description of model's attention focus.
        rsi: RSI indicator value.
        macd: MACD indicator value.
        ticker: Stock ticker symbol.

    Returns:
        Dictionary with:
            - accuracy: Score 1-5
            - clarity: Score 1-5
            - alignment: Score 1-5
            - average: Average score
            - feedback: LLM judge's comments
    """
    prompt = f"""You are an expert evaluator assessing the quality of AI-generated stock analysis explanations.

CONTEXT:
- Stock: {ticker}
- Model Prediction: {prediction} (confidence: {confidence:.1%})
- Model Attention: {attention_description if attention_description else 'N/A'}
- RSI: {rsi if rsi is not None else 'N/A'}
- MACD: {macd if macd is not None else 'N/A'}

EXPLANATION TO EVALUATE:
"{explanation}"

SCORING CRITERIA:
1. ACCURACY (1-5): Does the explanation correctly describe and support the prediction? Does it accurately reference the confidence level and data?
2. CLARITY (1-5): Is the explanation easy to understand for a retail investor with no technical background? Is jargon explained?
3. ALIGNMENT (1-5): Does the explanation reference relevant chart features and technical indicators? Does it mention what the model actually focused on?

RESPOND IN THIS EXACT FORMAT (JSON):
{{
    "accuracy": <score 1-5>,
    "clarity": <score 1-5>,
    "alignment": <score 1-5>,
    "feedback": "<brief feedback explaining the scores>"
}}

Return ONLY the JSON, no other text."""

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return _fallback_scores()

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)

        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=300,
            ),
        )

        # Parse the JSON response
        response_text = response.text.strip()

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
        else:
            scores = json.loads(response_text)

        # Validate scores
        for key in ["accuracy", "clarity", "alignment"]:
            if key not in scores:
                scores[key] = 3
            scores[key] = max(1, min(5, int(scores[key])))

        scores["average"] = round(
            (scores["accuracy"] + scores["clarity"] + scores["alignment"]) / 3, 2
        )

        if "feedback" not in scores:
            scores["feedback"] = "Evaluation completed."

        return scores

    except Exception as e:
        print(f"⚠️  LLM judge error: {e}")
        return _fallback_scores()


def _fallback_scores() -> dict:
    """
    Return default scores when the LLM judge is unavailable.

    Returns:
        Default scores dictionary.
    """
    return {
        "accuracy": 3,
        "clarity": 3,
        "alignment": 3,
        "average": 3.0,
        "feedback": "Fallback scoring used (LLM judge unavailable).",
    }


def batch_evaluate(
    evaluations: list[dict],
) -> dict:
    """
    Evaluate multiple explanations and compute aggregate scores.

    Args:
        evaluations: List of evaluation input dicts, each with:
            prediction, confidence, explanation, attention_description,
            rsi, macd, ticker.

    Returns:
        Dictionary with individual and aggregate scores.
    """
    results = []

    for i, eval_input in enumerate(evaluations):
        print(f"  Evaluating {i+1}/{len(evaluations)}...")
        scores = evaluate_explanation(
            prediction=eval_input.get("prediction", "Neutral"),
            confidence=eval_input.get("confidence", 0.5),
            explanation=eval_input.get("explanation", ""),
            attention_description=eval_input.get("attention_description", ""),
            rsi=eval_input.get("rsi"),
            macd=eval_input.get("macd"),
            ticker=eval_input.get("ticker", "STOCK"),
        )
        results.append(scores)

    # Compute aggregates
    if results:
        avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
        avg_clarity = sum(r["clarity"] for r in results) / len(results)
        avg_alignment = sum(r["alignment"] for r in results) / len(results)
        overall_avg = (avg_accuracy + avg_clarity + avg_alignment) / 3
    else:
        avg_accuracy = avg_clarity = avg_alignment = overall_avg = 0

    return {
        "individual_scores": results,
        "aggregate": {
            "accuracy": round(avg_accuracy, 2),
            "clarity": round(avg_clarity, 2),
            "alignment": round(avg_alignment, 2),
            "overall": round(overall_avg, 2),
            "num_evaluated": len(results),
        },
    }


def save_evaluation_report(
    evaluation_results: dict,
    output_dir: Path | None = None,
) -> Path:
    """
    Save LLM judge evaluation results to a JSON file.

    Args:
        evaluation_results: Results from batch_evaluate or evaluate_explanation.
        output_dir: Directory to save the report.

    Returns:
        Path to the saved report.
    """
    output_dir = output_dir or CHECKPOINTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": GEMINI_MODEL,
        **evaluation_results,
    }

    report_path = output_dir / "llm_judge_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📄 LLM judge report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    # Test evaluation
    print("🧑‍⚖️ Running LLM Judge Evaluation...")

    test_explanation = (
        "Based on the candlestick chart analysis, our AI model predicts an upward trend "
        "for AAPL with 78% confidence. The chart shows a series of consecutive green "
        "candles with increasing body sizes, suggesting growing buyer momentum. The RSI "
        "at 55 sits in a healthy neutral zone, indicating room for further upside. "
        "However, always consider this as one factor in your investment research."
    )

    scores = evaluate_explanation(
        prediction="Up",
        confidence=0.78,
        explanation=test_explanation,
        attention_description="Model focused on recent green candle bodies.",
        rsi=55.0,
        macd=1.25,
        ticker="AAPL",
    )

    print(f"\n{'='*50}")
    print(f"📊 Evaluation Scores:")
    print(f"   Accuracy:   {scores['accuracy']}/5")
    print(f"   Clarity:    {scores['clarity']}/5")
    print(f"   Alignment:  {scores['alignment']}/5")
    print(f"   Average:    {scores['average']}/5")
    print(f"\n   Feedback: {scores['feedback']}")
    print(f"{'='*50}")
