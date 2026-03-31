import os
import sys
import warnings

# Suppress noisy third-party deprecation warnings that we cannot fix
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', message='.*utcnow.*', category=FutureWarning)

# Ensure src/ modules are importable as bare names (matches main.py strategy)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data import MarketDataHandler, NewsFetcher
from sentiment import SentimentAgent

def test():
    print("Testing locally...")
    # 1. Market Data (yfinance fallback if no API keys)
    md = MarketDataHandler()
    df = md.get_historical_data("AAPL", start="2023-01-01", end="2023-01-10")
    print("\n[Market Data]")
    print(f"Data fetched successfully. Rows: {len(df)}")
    
    # 2. News API (Requires Alpaca keys)
    nf = NewsFetcher()
    if nf.api:
        news = nf.get_recent_news("AAPL", limit=2)
        print("\n[News Fetcher]")
        print(f"Fetched {len(news)} news items.")
        for n in news:
            print(f"- {n['headline']}")
    else:
        print("\n[News Fetcher]")
        print("Skipped: No Alpaca API keys found.")
        news = [{"headline": "Apple releases new iPhone.", "summary": "The new iPhone has a better camera."}]
        
    # 3. Sentiment Agent
    print("\n[Sentiment Agent]")
    agent = SentimentAgent()
    result = agent.analyze_news(news)
    # analyze_news returns a dict {'score': float, ...} or a plain float
    score_val = result['score'] if isinstance(result, dict) else result
    print(f"Sentiment Score for test news: {score_val:.2f}")
    if isinstance(result, dict):
        print(f"  Headlines used : {result.get('count', '?')}")
        print(f"  Sector         : {result.get('sector', 'unknown')}")

if __name__ == "__main__":
    test()
