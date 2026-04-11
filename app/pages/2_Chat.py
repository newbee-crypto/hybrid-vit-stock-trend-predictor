"""
2_Chat.py — AI Analyst Chat Page.

Provides a conversational interface with the LangChain analyst agent,
including memory, personalization, and suggested prompts.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

import streamlit as st

from config import STOCK_TICKERS
from utils.ui_helpers import inject_custom_css, section_header, custom_divider

inject_custom_css()

st.markdown("# 💬 AI Analyst Chat")
st.markdown("Chat with our AI stock analyst. Ask about trends, patterns, news, and more.")

st.markdown(custom_divider(), unsafe_allow_html=True)

# ━━━ Initialize ━━━
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory_manager" not in st.session_state:
    from src.memory.chat_memory import ChatMemoryManager
    st.session_state.memory_manager = ChatMemoryManager()

if "agent" not in st.session_state:
    st.session_state.agent = None


def initialize_agent():
    """Initialize the analyst agent."""
    if st.session_state.agent is None:
        try:
            from src.genai.analyst_agent import create_analyst_agent
            st.session_state.agent = create_analyst_agent()
            return True
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            return False
    return True


# ━━━ Sidebar: Session Info ━━━
with st.sidebar:
    st.markdown(section_header("📊 Session Info"), unsafe_allow_html=True)

    memory = st.session_state.memory_manager
    stats = memory.get_stats()

    st.metric("Chat Turns", stats["total_turns"])
    st.metric("Stocks Queried", stats["unique_stocks"])
    st.metric("Session (min)", stats["session_minutes"])

    if stats["interested_stocks"]:
        st.markdown("**🎯 Your Interests:**")
        for stock in stats["interested_stocks"]:
            count = stats["stocks_queried"][stock]
            st.markdown(f"  • {stock} ({count} queries)")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory_manager.clear()
        st.session_state.agent = None
        st.rerun()

# ━━━ Suggested Prompts ━━━
if not st.session_state.chat_history:
    st.markdown(section_header("💡 Suggested Prompts"), unsafe_allow_html=True)

    suggestions = [
        f"Analyze the latest chart for {STOCK_TICKERS[0]}",
        f"What's the news sentiment for {STOCK_TICKERS[3]}?",
        f"Compare {STOCK_TICKERS[0]} and {STOCK_TICKERS[2]} technical indicators",
        f"Give me the full signal fusion analysis for {STOCK_TICKERS[5]}",
        "Which stock has the most bullish pattern right now?",
        "Explain what RSI, MACD, and trend score mean in simple terms",
    ]

    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(f"💡 {suggestion[:40]}...", key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_message = suggestion
                st.rerun()

# ━━━ Chat History ━━━
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["human"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(turn["ai"])

# ━━━ Chat Input ━━━
# Check for pending message from suggestions
pending = st.session_state.pop("pending_message", None)
user_input = st.chat_input("Ask the analyst anything...") or pending

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize agent if needed
    if not initialize_agent():
        st.stop()

    # Get response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Thinking..."):
            # Get personalization message
            personalization = st.session_state.memory_manager.get_personalization_message()

            # Get chat history for context
            history = st.session_state.memory_manager.get_chat_history()

            try:
                from src.genai.analyst_agent import chat_with_agent

                # Prepend personalization context if available
                context_prefix = ""
                if personalization:
                    context_prefix = f"[Context for you: {st.session_state.memory_manager.get_personalized_context()}] "

                response = chat_with_agent(
                    st.session_state.agent,
                    context_prefix + user_input,
                    history,
                )

                # Clean up context prefix from response if it leaked
                if response.startswith("[Context"):
                    response = response.split("]", 1)[-1].strip()

                st.markdown(response)

            except Exception as e:
                response = f"I apologize, I encountered an error: {str(e)}. Please try again."
                st.error(response)

    # Save to memory
    st.session_state.chat_history.append({
        "human": user_input,
        "ai": response,
    })
    st.session_state.memory_manager.add_turn(user_input, response)
