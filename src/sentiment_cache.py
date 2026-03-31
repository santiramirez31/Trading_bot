"""
sentiment_cache.py -- Historical sentiment pre-computation and caching

For each trading day in the backtest window, fetches real Alpaca news
headlines and scores them with SentimentAgent. Results are cached to a
CSV file so subsequent backtest runs load instantly without re-running
LLM inference.

Cache location: data/sentiment_cache_{TICKER}.csv
Cache columns : date, score, consensus_pct, headline_count

Usage (from backtest.py):
    from sentiment_cache import get_sentiment_series
    scores = get_sentiment_series(ticker, trading_dates,
                                  sentiment_agent=agent,
                                  news_fetcher=fetcher)
    df['sentiment'] = scores
"""

import os
import time
import math
import pandas as pd
from datetime import timedelta, datetime, date as date_type

# Cache lives in data/ at project root (one level above src/)
_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
NEUTRAL_SCORE = 5.0

# ── Jaccard deduplication parameters (mirrors colleague spec) ──────────────
_JACCARD_N         = 3      # character n-gram size
_JACCARD_THRESHOLD = 0.72   # similarity above which an article is a near-duplicate

# ── Time-decay weighting parameters ───────────────────────────────────────
_DECAY_LAMBDA = 0.08        # per-hour decay rate: weight = exp(-λ × age_hours)
                             # → 8h old  ≈ 0.53×,  20h old ≈ 0.20×,  48h old ≈ 0.02×


def _ngrams(text: str, n: int = _JACCARD_N) -> set:
    """Return the set of character n-grams for a lowercased string."""
    t = text.lower()
    return set(t[i:i+n] for i in range(len(t) - n + 1)) if len(t) >= n else set()


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two strings based on character n-grams."""
    sa, sb = _ngrams(a), _ngrams(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(articles: list) -> list:
    """
    Remove near-duplicate headlines using greedy Jaccard filtering.
    First occurrence is kept; later articles with similarity >= threshold are dropped.
    This prevents syndicated copies of the same story from inflating counts/consensus.
    """
    kept = []
    for art in articles:
        h = art.get('headline', '')
        if any(_jaccard(h, k.get('headline', '')) >= _JACCARD_THRESHOLD for k in kept):
            continue
        kept.append(art)
    return kept


def _time_decay_weight(published_at, reference_dt) -> float:
    """
    Exponential decay weight based on article age.
    weight = exp(-DECAY_LAMBDA × age_in_hours)
    If published_at is unavailable, returns 1.0 (no decay).
    """
    if published_at is None:
        return 1.0
    try:
        if isinstance(published_at, str):
            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        if reference_dt.tzinfo is None:
            reference_dt = reference_dt.replace(tzinfo=published_at.tzinfo)
        age_h = max((reference_dt - published_at).total_seconds() / 3600, 0)
        return math.exp(-_DECAY_LAMBDA * age_h)
    except Exception:
        return 1.0


def _cache_path(ticker: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'sentiment_cache_{ticker.upper()}.csv')


def _load_cache(ticker: str, cache_dir: str) -> pd.DataFrame:
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).date
        # backward-compat: old caches may lack momentum column
        if 'momentum' not in df.columns:
            df['momentum'] = 0
        return df
    return pd.DataFrame(columns=['score', 'consensus_pct', 'headline_count', 'momentum'])


def _fetch_news_for_date(api, ticker: str, day: date_type) -> list:
    """
    Fetch Alpaca news headlines for a single trading day.
    Returns dicts including published_at for time-decay weighting.
    """
    start_str = day.strftime('%Y-%m-%dT00:00:00Z')
    end_str   = (day + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    try:
        raw = api.get_news(ticker, start=start_str, end=end_str, limit=30)
        articles = []
        for n in raw:
            pub = getattr(n, 'created_at', None) or getattr(n, 'published_at', None)
            articles.append({
                'headline':     n.headline,
                'summary':      getattr(n, 'summary', ''),
                'published_at': pub,
            })
        return articles
    except Exception:
        return []


def build_or_update_cache(
    ticker: str,
    trading_dates: list,
    sentiment_agent,
    news_fetcher,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """
    Pre-computes real LLM sentiment for every trading day not already cached.
    Appends results to the cache CSV and returns the complete DataFrame.
    """
    cache_df = _load_cache(ticker, cache_dir)
    cached   = set(cache_df.index.tolist()) if not cache_df.empty else set()
    missing  = [d for d in trading_dates if d not in cached]

    if not missing:
        print(f"  [CACHE] {ticker}: all {len(trading_dates)} dates already cached.")
        return cache_df

    total = len(missing)
    print(f"  [CACHE] {ticker}: computing sentiment for {total} trading days "
          f"(~{total * sleep_sec / 60:.1f} min)...")

    if not news_fetcher.api:
        print("  [CACHE] Alpaca API not available -- cannot fetch historical news.")
        return cache_df

    new_rows = []
    for i, day in enumerate(missing):
        raw_articles  = _fetch_news_for_date(news_fetcher.api, ticker, day)

        # Step 1: Jaccard deduplication — remove syndicated copies before scoring
        deduped = _deduplicate(raw_articles)
        n_dropped = len(raw_articles) - len(deduped)

        if deduped:
            # Step 2: Time-decay weights — embed on each article so the weight
            # travels through any further filtering inside analyze_news.
            # Use end-of-day as reference so intraday ordering is preserved.
            ref_dt = datetime(day.year, day.month, day.day, 23, 59, 59)
            for art in deduped:
                art['_decay_weight'] = _time_decay_weight(art.get('published_at'), ref_dt)

            result = sentiment_agent.analyze_news(
                deduped, ticker=ticker, verbose=False
            )
            row = {
                'date':           day,
                'score':          result['score'],
                'consensus_pct':  result['consensus_pct'],
                'headline_count': result.get('article_count', result.get('headline_count', 0)),
                'momentum':       result.get('momentum', 0),
            }
            if n_dropped:
                pass  # silently swallow; only log in verbose mode
        else:
            row = {'date': day, 'score': NEUTRAL_SCORE, 'consensus_pct': 0.0,
                   'headline_count': 0, 'momentum': 0}

        new_rows.append(row)

        # Progress every 20 days
        if (i + 1) % 20 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            print(f"  [CACHE] {ticker}: {i+1}/{total} days done ({pct:.0f}%)  "
                  f"last score={row['score']:.1f}")

        time.sleep(sleep_sec)

    # Merge with existing cache and persist
    new_df   = pd.DataFrame(new_rows).set_index('date')
    combined = pd.concat([cache_df, new_df]).sort_index()
    combined.reset_index().to_csv(_cache_path(ticker, cache_dir), index=False)
    print(f"  [CACHE] {ticker}: saved {len(combined)} days to cache.")
    return combined


def get_sentiment_series(
    ticker: str,
    trading_dates: list,
    sentiment_agent=None,
    news_fetcher=None,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> list:
    """
    Returns a list of sentiment scores (float, 0-10) aligned to trading_dates.

    - If sentiment_agent + news_fetcher are provided: builds/updates cache first.
    - If cache is missing for a date: uses NEUTRAL_SCORE (5.0) as fallback.
    """
    if sentiment_agent is not None and news_fetcher is not None:
        cache_df = build_or_update_cache(
            ticker, trading_dates, sentiment_agent, news_fetcher, cache_dir
        )
    else:
        cache_df = _load_cache(ticker, cache_dir)

    scores = []
    for d in trading_dates:
        if not cache_df.empty and d in cache_df.index:
            scores.append(float(cache_df.loc[d, 'score']))
        else:
            scores.append(NEUTRAL_SCORE)

    cached_count  = sum(1 for s in scores if s != NEUTRAL_SCORE)
    neutral_count = len(scores) - cached_count
    print(f"  [CACHE] {ticker}: {cached_count} days with real scores, "
          f"{neutral_count} days defaulted to neutral.")
    return scores
