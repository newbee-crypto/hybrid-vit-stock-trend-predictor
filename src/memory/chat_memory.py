"""
chat_memory.py — Conversation memory and user interest tracking.

Manages chat history with a sliding window, tracks frequently queried
stocks, and personalizes responses based on user interests.

Usage:
    from src.memory.chat_memory import ChatMemoryManager
    memory = ChatMemoryManager()
    memory.add_turn("What about TSLA?", "TSLA shows...")
    context = memory.get_personalized_context()
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import re
from collections import Counter
from datetime import datetime

from config import MEMORY_WINDOW, INTEREST_THRESHOLD, STOCK_TICKERS


class ChatMemoryManager:
    """
    Manages conversation memory with sliding window and interest tracking.

    Keeps the last K conversation turns and tracks which stocks the user
    queries most frequently to personalize responses.

    Attributes:
        max_turns: Maximum number of turns to remember.
        history: List of (human, ai, timestamp) tuples.
        stock_queries: Counter of stock ticker mentions.
        session_start: When this session started.
    """

    def __init__(self, max_turns: int = MEMORY_WINDOW):
        """
        Initialize the memory manager.

        Args:
            max_turns: Maximum conversation turns to keep in memory.
        """
        self.max_turns = max_turns
        self.history: list[dict] = []
        self.stock_queries: Counter = Counter()
        self.session_start = datetime.now()
        self.user_preferences: dict = {}

    def add_turn(self, human_message: str, ai_response: str) -> None:
        """
        Add a new conversation turn to memory.

        Automatically trims old turns beyond the window size.

        Args:
            human_message: The user's message.
            ai_response: The AI's response.
        """
        turn = {
            "human": human_message,
            "ai": ai_response,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(turn)

        # Trim to window size
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

        # Track stock mentions
        mentioned_tickers = self._extract_tickers(human_message)
        for ticker in mentioned_tickers:
            self.stock_queries[ticker] += 1

    def get_chat_history(self) -> list[tuple[str, str]]:
        """
        Get the conversation history as (human, ai) pairs.

        Returns:
            List of (human_message, ai_response) tuples.
        """
        return [(turn["human"], turn["ai"]) for turn in self.history]

    def get_history_text(self) -> str:
        """
        Get the conversation history as formatted text.

        Returns:
            Formatted string of the conversation history.
        """
        if not self.history:
            return "No conversation history yet."

        lines = []
        for turn in self.history:
            lines.append(f"User: {turn['human']}")
            lines.append(f"Analyst: {turn['ai'][:200]}...")
            lines.append("")

        return "\n".join(lines)

    def get_interested_stocks(self) -> list[str]:
        """
        Get stocks the user has shown interest in (queried >= threshold times).

        Returns:
            List of ticker symbols the user frequently queries.
        """
        return [
            ticker for ticker, count in self.stock_queries.most_common()
            if count >= INTEREST_THRESHOLD
        ]

    def get_top_stock(self) -> str | None:
        """
        Get the most frequently queried stock ticker.

        Returns:
            Most queried ticker, or None if no queries recorded.
        """
        if self.stock_queries:
            return self.stock_queries.most_common(1)[0][0]
        return None

    def get_personalized_context(self) -> str:
        """
        Generate personalized context based on user's query history.

        Returns insights about the user's interests to help the AI
        provide more relevant responses.

        Returns:
            Personalized context string.
        """
        context_parts = []

        # Session info
        session_duration = (datetime.now() - self.session_start).seconds // 60
        context_parts.append(f"Session duration: {session_duration} minutes")

        # Interested stocks
        interested = self.get_interested_stocks()
        if interested:
            context_parts.append(
                f"The user has shown particular interest in: {', '.join(interested)}. "
                f"Consider providing more detailed analysis for these stocks."
            )

        # Most queried stock
        top_stock = self.get_top_stock()
        if top_stock:
            count = self.stock_queries[top_stock]
            if count >= INTEREST_THRESHOLD:
                context_parts.append(
                    f"The user has asked about {top_stock} {count} times this session. "
                    f"They may be considering a position in {top_stock}."
                )

        # Query patterns
        total_queries = sum(self.stock_queries.values())
        if total_queries > 5:
            unique_stocks = len(self.stock_queries)
            if unique_stocks == 1:
                context_parts.append(
                    "The user is focused on a single stock — provide deep analysis."
                )
            elif unique_stocks > 3:
                context_parts.append(
                    "The user is researching multiple stocks — they may be building a diversified portfolio."
                )

        # Previous conversation context
        if self.history:
            last_turn = self.history[-1]
            context_parts.append(
                f"Last question was: '{last_turn['human'][:100]}'"
            )

        return " | ".join(context_parts) if context_parts else ""

    def get_personalization_message(self) -> str | None:
        """
        Generate a personalization message to prepend to AI responses.

        Returns:
            Personalization message, or None if no personalization needed.
        """
        interested = self.get_interested_stocks()
        if not interested:
            return None

        top = interested[0]
        count = self.stock_queries[top]

        messages = {
            3: f"I notice you've been watching {top} closely. ",
            5: f"Since you've been particularly interested in {top}, ",
            7: f"As a frequent {top} follower, you might want to know ",
        }

        for threshold in sorted(messages.keys(), reverse=True):
            if count >= threshold:
                return messages[threshold]

        return None

    def _extract_tickers(self, text: str) -> list[str]:
        """
        Extract stock ticker symbols from a text string.

        Args:
            text: Input text to search for tickers.

        Returns:
            List of found ticker symbols.
        """
        # Match known tickers
        found = []
        text_upper = text.upper()

        for ticker in STOCK_TICKERS:
            # Match whole word only
            if re.search(rf'\b{ticker}\b', text_upper):
                found.append(ticker)

        return found

    def clear(self) -> None:
        """Clear all conversation history and tracking data."""
        self.history.clear()
        self.stock_queries.clear()
        self.session_start = datetime.now()
        self.user_preferences.clear()

    def get_stats(self) -> dict:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory stats.
        """
        return {
            "total_turns": len(self.history),
            "max_turns": self.max_turns,
            "stocks_queried": dict(self.stock_queries),
            "unique_stocks": len(self.stock_queries),
            "total_queries": sum(self.stock_queries.values()),
            "session_minutes": (datetime.now() - self.session_start).seconds // 60,
            "interested_stocks": self.get_interested_stocks(),
        }


if __name__ == "__main__":
    # Test the memory manager
    memory = ChatMemoryManager(max_turns=5)

    # Simulate a conversation
    test_turns = [
        ("What's the trend for AAPL?", "AAPL shows an upward trend..."),
        ("How about TSLA?", "TSLA appears to be declining..."),
        ("Show me AAPL again", "AAPL still shows bullish signals..."),
        ("Any news on AAPL?", "Recent AAPL news is positive..."),
        ("AAPL technical indicators?", "AAPL RSI is at 55..."),
    ]

    for human, ai in test_turns:
        memory.add_turn(human, ai)
        print(f"Added: {human}")

    print(f"\n{'='*50}")
    print(f"📊 Memory Stats:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print(f"\n🎯 Personalized Context:")
    print(f"   {memory.get_personalized_context()}")

    print(f"\n💬 Personalization Message:")
    print(f"   {memory.get_personalization_message()}")
