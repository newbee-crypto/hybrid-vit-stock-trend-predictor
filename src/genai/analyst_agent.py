"""
analyst_agent.py — LangChain-powered stock analyst agent with tools.

Provides a conversational AI agent that can analyze charts, search news,
find patterns, compute signals, and explain results using Gemini.

Usage:
    from src.genai.analyst_agent import create_analyst_agent, chat_with_agent
    agent = create_analyst_agent()
    response = chat_with_agent(agent, "Analyze AAPL chart for me")
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from config import GEMINI_MODEL

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━


@tool
def analyze_chart(ticker: str) -> str:
    """
    Analyze a stock's recent candlestick chart using the ViT model.
    Returns the predicted trend direction and confidence.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    """
    try:
        from src.model.inference import predict_image
        from config import CHARTS_DIR, LABELS_DIR
        import pandas as pd

        chart_dir = CHARTS_DIR / ticker
        if not chart_dir.exists():
            return f"No chart data available for {ticker}. Please run data pipeline first."

        # Get the most recent chart
        charts = sorted(chart_dir.glob("*.png"))
        if not charts:
            return f"No chart images found for {ticker}."

        latest_chart = charts[-1]
        aux_features = None
        labels_path = LABELS_DIR / "labeled_dataset.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            ticker_data = df[df["ticker"] == ticker].sort_values("date")
            if not ticker_data.empty:
                last_row = ticker_data.iloc[-1]
                aux_features = [
                    float(last_row.get("RSI", 0.0) or 0.0),
                    float(last_row.get("MACD", 0.0) or 0.0),
                    float(last_row.get("trend_score", 0.0) or 0.0),
                ]

        result = predict_image(str(latest_chart), aux_features=aux_features)

        response = (
            f"Chart Analysis for {ticker}:\n"
            f"- Trend: {result['prediction']}\n"
            f"- Confidence: {result['confidence']:.1%}\n"
            f"- Probabilities: Up={result['probabilities']['Up']:.1%}, "
            f"Neutral={result['probabilities']['Neutral']:.1%}, "
            f"Down={result['probabilities']['Down']:.1%}\n"
        )

        if result.get("is_fallback"):
            response += "⚠️ Note: Using untrained model (results may not be reliable)\n"

        return response

    except Exception as e:
        return f"Error analyzing chart for {ticker}: {str(e)}"


@tool
def get_news(ticker: str) -> str:
    """
    Search for recent news about a stock and analyze sentiment.
    Returns a summary with sentiment score.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    """
    try:
        from src.genai.signal_fusion import get_news_signal

        result = get_news_signal(ticker)

        sentiment = "Positive" if result["score"] > 0.1 else "Negative" if result["score"] < -0.1 else "Neutral"

        response = (
            f"News Sentiment for {ticker}:\n"
            f"- Sentiment: {sentiment} (score: {result['score']:+.2f})\n"
            f"- Signal: {result['signal']}\n"
            f"- Confidence: {result['confidence']:.1%}\n"
            f"- Articles analyzed: {result.get('num_articles', 'N/A')}\n"
            f"- Summary: {result.get('summary', 'N/A')}\n"
        )
        return response

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


@tool
def find_patterns(prediction: str, confidence: float) -> str:
    """
    Find similar historical candlestick patterns using RAG.
    Returns the top 2 matching patterns.

    Args:
        prediction: Current trend prediction ('Up', 'Down', or 'Neutral')
        confidence: Prediction confidence as a float (e.g., 0.78)
    """
    try:
        from src.genai.rag_patterns import find_similar_patterns, initialize_pattern_db

        # Ensure DB is initialized
        initialize_pattern_db()

        patterns = find_similar_patterns(
            prediction=prediction,
            confidence=confidence,
        )

        if not patterns:
            return "No similar patterns found in the database."

        response = "Similar Candlestick Patterns:\n"
        for i, p in enumerate(patterns, 1):
            response += (
                f"\n{i}. {p['name']} ({p['signal']})\n"
                f"   {p['description']}\n"
                f"   Similarity: {p['similarity_score']:.0%}, "
                f"Historical accuracy: {p['historical_accuracy']:.0%}\n"
            )

        return response

    except Exception as e:
        return f"Error finding patterns: {str(e)}"


@tool
def get_fusion_signal(ticker: str) -> str:
    """
    Compute the combined signal fusion for a stock.
    Combines ViT prediction, RAG patterns, news sentiment, and technical indicators.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    """
    try:
        from src.genai.signal_fusion import fuse_signals
        from src.model.inference import predict_image
        from src.genai.rag_patterns import find_similar_patterns, initialize_pattern_db
        from config import CHARTS_DIR, LABELS_DIR
        import pandas as pd

        # Get ViT prediction
        chart_dir = CHARTS_DIR / ticker
        vit_result = None
        aux_features = None
        if chart_dir.exists():
            charts = sorted(chart_dir.glob("*.png"))
            labels_path = LABELS_DIR / "labeled_dataset.csv"
            if labels_path.exists():
                df = pd.read_csv(labels_path)
                ticker_data = df[df["ticker"] == ticker].sort_values("date")
                if not ticker_data.empty:
                    last_row = ticker_data.iloc[-1]
                    aux_features = [
                        float(last_row.get("RSI", 0.0) or 0.0),
                        float(last_row.get("MACD", 0.0) or 0.0),
                        float(last_row.get("trend_score", 0.0) or 0.0),
                    ]
                    rsi = last_row.get("RSI")
                    macd = last_row.get("MACD")
                    macd_sig = last_row.get("MACD_signal")
                else:
                    rsi, macd, macd_sig = None, None, None
            else:
                rsi, macd, macd_sig = None, None, None

            if charts:
                vit_result = predict_image(str(charts[-1]), aux_features=aux_features)

        # Get RAG patterns
        initialize_pattern_db()
        pred = vit_result.get("prediction", "Neutral") if vit_result else "Neutral"
        conf = vit_result.get("confidence", 0.5) if vit_result else 0.5
        patterns = find_similar_patterns(pred, conf)

        # Fuse signals
        result = fuse_signals(
            vit_result=vit_result,
            rag_patterns=patterns,
            ticker=ticker,
            rsi=rsi,
            macd=macd,
            macd_signal_val=macd_sig,
        )

        response = (
            f"Signal Fusion for {ticker}:\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Final Signal: {result['signal']}\n"
            f"Confidence: {result['confidence']:.1%}\n"
            f"Combined Score: {result['combined_score']:+.3f}\n"
            f"Signal Agreement: {result['agreement']:.0%}\n\n"
            f"Component Breakdown:\n"
        )

        for name, sig in result["components"].items():
            weight = result["weights"][name]
            response += f"  {name:>10} ({weight:.0%}): {sig['signal']} (score: {sig['score']:+.3f})\n"

        return response

    except Exception as e:
        return f"Error computing fusion signal for {ticker}: {str(e)}"


@tool
def get_technicals(ticker: str) -> str:
    """
    Get RSI, MACD, and trend score values for a stock.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
    """
    try:
        from config import LABELS_DIR
        import pandas as pd

        labels_path = LABELS_DIR / "labeled_dataset.csv"
        if not labels_path.exists():
            return f"No labeled data available for {ticker}. Run data pipeline first."

        df = pd.read_csv(labels_path)
        ticker_data = df[df["ticker"] == ticker].sort_values("date")

        if ticker_data.empty:
            return f"No data found for {ticker}."

        last_row = ticker_data.iloc[-1]

        rsi = last_row.get("RSI", "N/A")
        macd = last_row.get("MACD", "N/A")
        trend_score = last_row.get("trend_score", "N/A")
        macd_sig = last_row.get("MACD_signal", "N/A")

        rsi_interpretation = ""
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                rsi_interpretation = " (Overbought ⚠️)"
            elif rsi < 30:
                rsi_interpretation = " (Oversold 📉)"
            elif rsi > 50:
                rsi_interpretation = " (Bullish bias)"
            else:
                rsi_interpretation = " (Bearish bias)"

        rsi_text = f"{float(rsi):.2f}" if isinstance(rsi, (int, float)) else "N/A"
        macd_text = f"{float(macd):.4f}" if isinstance(macd, (int, float)) else "N/A"
        trend_text = f"{float(trend_score):.4f}" if isinstance(trend_score, (int, float)) else "N/A"
        macd_sig_text = f"{float(macd_sig):.4f}" if isinstance(macd_sig, (int, float)) else "N/A"
        macd_cross = "Bullish ↑" if isinstance(macd, (int, float)) and isinstance(macd_sig, (int, float)) and macd > macd_sig else "Bearish ↓"

        response = (
            f"Technical Indicators for {ticker} (as of {last_row.get('date', 'N/A')}):\n"
            f"- RSI (14): {rsi_text}{rsi_interpretation}\n"
            f"- MACD: {macd_text}\n"
            f"- Trend Score: {trend_text}\n"
            f"- MACD Signal: {macd_sig_text}\n"
            f"- MACD Cross: {macd_cross}\n"
        )
        return response

    except Exception as e:
        return f"Error getting technicals for {ticker}: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT CREATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """You are a professional stock market analyst AI assistant. Your role is to help investors understand stock trends using candlestick chart analysis, technical indicators, and market news.

KEY BEHAVIORS:
1. Always be helpful, clear, and thorough in your analysis
2. Use the available tools to gather data before making recommendations
3. Explain technical concepts in simple terms
4. Always include appropriate risk disclaimers
5. Reference specific data points (RSI values, confidence scores, etc.)
6. If you notice the user frequently asks about a stock, acknowledge their interest
7. Be proactive: suggest related analyses the user might find useful

ANALYSIS WORKFLOW:
When asked to analyze a stock:
1. First use analyze_chart to get the ViT prediction
2. Then use find_patterns to find similar historical patterns
3. Use get_news for current sentiment
4. Use get_technicals for RSI/MACD/trend score
5. Optionally use get_fusion_signal for the combined signal
6. Synthesize everything into a clear, actionable summary

DISCLAIMER: Always remind users that AI predictions are for educational purposes and should not be the sole basis for investment decisions."""


def create_analyst_agent(
    memory=None,
    verbose: bool = False,
) -> AgentExecutor:
    """
    Create a LangChain analyst agent with chart analysis tools.

    Args:
        memory: LangChain memory instance for conversation history.
        verbose: Whether to show agent's reasoning steps.

    Returns:
        Configured AgentExecutor ready for conversation.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment.")

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.7,
        max_output_tokens=1024,
    )

    # Define tools
    tools = [
        analyze_chart,
        get_news,
        find_patterns,
        get_fusion_signal,
        get_technicals,
    ]

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=False,
    )

    return executor


def chat_with_agent(
    agent: AgentExecutor,
    message: str,
    chat_history: list | None = None,
) -> str:
    """
    Send a message to the analyst agent and get a response.

    Args:
        agent: The AgentExecutor instance.
        message: User's message/question.
        chat_history: List of previous (human, ai) message pairs.

    Returns:
        Agent's response string.
    """
    chat_history = chat_history or []

    # Convert chat history to LangChain message format
    history_messages = []
    for human_msg, ai_msg in chat_history:
        history_messages.append(HumanMessage(content=human_msg))
        history_messages.append(AIMessage(content=ai_msg))

    try:
        result = agent.invoke({
            "input": message,
            "chat_history": history_messages,
        })
        return result.get("output", "I apologize, I couldn't generate a response.")

    except Exception as e:
        return (
            f"I encountered an error while processing your request: {str(e)}. "
            f"Please try again or rephrase your question."
        )


if __name__ == "__main__":
    print("🤖 Starting Analyst Agent...")
    agent = create_analyst_agent(verbose=True)

    print("\n💬 Chat with the analyst (type 'quit' to exit):\n")
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if not user_input:
            continue

        response = chat_with_agent(agent, user_input, history)
        print(f"\n🤖 Analyst: {response}\n")
        history.append((user_input, response))
