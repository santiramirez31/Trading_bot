"""
backtest_portfolio.py -- Portfolio-Level Backtest Using the Full 5-Criteria Pipeline

Mirrors the live bot exactly:
  1. Runs the S&P 500 5-criteria funnel (C1+C4/C5) as of the backtest start date
     (no look-ahead bias -- historical data only up to that date).
  2. Pre-builds LLM sentiment caches for all survivors over the backtest window.
  3. Simulates a 10-position portfolio day-by-day:
       - Entry: top candidates by rank_score that pass C3 sentiment (final_score>=0.65,
                consensus>=70%, momentum>=0, headlines>=5)
       - Exit:  2% stop-loss (bracket) or 5% take-profit
       - Replace: when a position closes, scan ranked candidates in order, apply C3
                  using cached sentiment for that date, buy the first that passes.
  4. Outputs:
       - plots/portfolio_equity_curve.png  (portfolio vs SPY buy-and-hold)
       - plots/portfolio_trades.csv        (full trade log)
       - Console summary with return, Sharpe, max drawdown, trades breakdown

Usage:
    python src/backtest_portfolio.py
    python src/backtest_portfolio.py --backtest-start 2025-03-26
    python src/backtest_portfolio.py --backtest-start 2025-03-26 --skip-cache-build
    python src/backtest_portfolio.py --tickers AAPL,MSFT,NVDA,GOOGL,META,AMZN,NFLX,AMD,CRM,TSLA
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src/ to path so imports work when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from data import MarketDataHandler, NewsFetcher
from sentiment import SentimentAgent
from sentiment_cache import build_or_update_cache, _load_cache, NEUTRAL_SCORE
from industry_weights import TICKER_SECTOR_MAP, normalize_sector

# -- Constants (mirror bot.py) ----------------------------------------------
INITIAL_CAPITAL     = 100_000.0
STOP_LOSS_PCT       = 0.05        # fallback only; ATR stop is used when available
ATR_MULTIPLIER      = 3.5         # stop = peak_price - ATR(14) * ATR_MULTIPLIER  [OOS validated]
TAKE_PROFIT_PCT     = 0.12        # OOS validated — lower risk than 20%
MAX_OPEN_POSITIONS  = 10
REENTRY_COOLDOWN_DAYS = 5       # trading days before re-entering the same ticker
RISK_FREE_RATE      = 0.04
# Risk-based position sizing (replaces flat POSITION_SIZE_PCT / MAX_SHARES)
RISK_PER_TRADE_PCT  = 0.02   # risk 2% of current portfolio per trade
MAX_POSITION_PCT    = 0.15   # single position capped at 15% of cash
                              # (2% risk / 5% stop = 40% sizing; cap brings it to 15%
                              #  so up to ~6 positions can be held simultaneously)
SENTIMENT_FLOOR       = 6.0   # LLM score must be at least this (out of 10)  [grid best]
TECH_FLOOR            = 0.3   # tech_signal_norm floor (not in a downtrend)
CONSENSUS_THRESHOLD   = 0.70
MIN_HEADLINES         = 5
SENTIMENT_SELL_FLOOR  = 4.0   # sell if sentiment drops below this...
SENTIMENT_SELL_CONV   = 0.70  # ...with at least this consensus conviction
RSI_ENTRY_MIN         = 35    # RSI floor: don't buy into a falling knife (oversold/broken)
TREND_FAIL_MIN_HOLD   = 5     # minimum trading days held before TREND_FAIL can trigger
                               # prevents RSI-dip entries from immediately exiting on MA50
_DEFAULT_CACHE_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')


# -- Data structures --------------------------------------------------------
@dataclass
class Position:
    ticker:       str
    entry_date:   date
    entry_price:  float
    shares:       int
    stop_price:     float          # trailing: peak_price - atr_stop_dist, updated daily
    target_price:   float          # entry * (1 + TAKE_PROFIT_PCT), fixed
    rank_index:     int            # position in the original ranked list
    peak_price:     float = 0.0   # highest close seen since entry (for trailing stop)
    atr_stop_dist:  float = 0.0   # ATR(14) * ATR_MULTIPLIER at entry — fixed trail distance

@dataclass
class ClosedTrade:
    ticker:       str
    entry_date:   date
    entry_price:  float
    shares:       int
    exit_date:    date
    exit_price:   float
    exit_reason:  str            # 'STOP_LOSS' | 'TAKE_PROFIT' | 'OPEN'
    rank_at_entry: int


# -- Step 1: Historical pipeline --------------------------------------------
def run_historical_pipeline(backtest_start: str, inflation_rate: float = 0.035,
                            ticker_override: list = None):
    """
    Runs C1 + C4/C5 on the full S&P 500 universe using data AS OF backtest_start.
    No look-ahead: yfinance downloads are capped at backtest_start.
    C3 sentiment is NOT applied here -- it is applied per-day during simulation.

    ticker_override: if provided, skip the S&P 500 universe fetch and screen
                     only these tickers through C1/C4/C5.  Useful for testing
                     the sentiment layer with high-news-volume stocks.

    Returns (ranked_candidates: list, sectors: dict).
    """
    md = MarketDataHandler()

    print(f"\n{'='*65}")
    print(f"  [Step 1/3] Historical pipeline as of {backtest_start}")
    print(f"{'='*65}")

    if ticker_override:
        tickers  = ticker_override
        sectors  = {t: 'Override' for t in tickers}
        print(f"  Ticker override: {tickers}")
    else:
        tickers, sectors = md.get_sp500_universe()
        if not tickers:
            print("  Failed to fetch S&P 500 universe.")
            sys.exit(1)

    quality = md.quality_screen_10yr(
        tickers,
        inflation_rate=inflation_rate,
        min_excess_pct=0.05,
        as_of_date=backtest_start,
    )
    if not quality:
        print("  No tickers passed C1. Exiting.")
        sys.exit(0)

    ranked = md.screen_liquidity_risk_trend(
        quality,
        as_of_date=backtest_start,
    )
    if not ranked:
        print("  No tickers passed C4/C5. Exiting.")
        sys.exit(0)

    print(f"\n  Top 15 ranked candidates (as of {backtest_start}):")
    print(f"  {'TICKER':<6} {'RANK':>5} {'TECH':>6} {'RISK':>6} {'10YR%':>6} {'SHARPE':>6} {'VOL%':>5} {'DD%':>6}")
    print(f"  {'-'*6} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")
    for c in ranked[:15]:
        print(f"  {c['ticker']:<6} {c['rank_score']:>5.3f} {c['tech_score']:>6.3f} "
              f"{c['risk_score']:>6.3f} {c['ann_return']*100:>6.1f} {c['sharpe']:>6.2f} "
              f"{c['volatility']*100:>5.1f} {c['max_drawdown']*100:>6.1f}")

    return ranked, sectors


# -- Step 2: Prefetch price data --------------------------------------------
def prefetch_prices(tickers: list, start: str, end: str, ma_warmup_days: int = 250) -> dict:
    """
    Batch-downloads OHLCV for all candidate tickers + SPY.
    Returns dict: ticker -> DataFrame(Open, High, Low, Close, Volume).
    Tickers with no data are excluded silently.

    ma_warmup_days: extra calendar days before `start` to include so that
                    200-day moving averages are accurate on the first simulation day.
    """
    all_tickers = list(dict.fromkeys(tickers + ['SPY']))  # deduplicate, preserve order
    # Extend the download window backwards so MAs are properly warmed up
    dl_start = (pd.Timestamp(start) - pd.Timedelta(days=ma_warmup_days + 100)).strftime('%Y-%m-%d')
    print(f"\n  Downloading price data for {len(all_tickers)} tickers ({start} -> {end})...")
    print(f"  (Fetching from {dl_start} for MA warm-up)")
    raw = yf.download(all_tickers, start=dl_start, end=end, progress=False, auto_adjust=True)

    price_data = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in all_tickers:
            try:
                df = pd.DataFrame({
                    'Open':   raw['Open'][ticker],
                    'High':   raw['High'][ticker],
                    'Low':    raw['Low'][ticker],
                    'Close':  raw['Close'][ticker],
                    'Volume': raw['Volume'][ticker],
                }).dropna()
                if len(df) > 10:
                    price_data[ticker] = df
            except Exception:
                continue
    else:
        # Single ticker (shouldn't happen in practice)
        ticker = all_tickers[0]
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        if len(df) > 10:
            price_data[ticker] = df

    available = [t for t in all_tickers if t in price_data]
    missing   = [t for t in tickers if t not in price_data]
    print(f"  {len(available)} tickers with data. {len(missing)} skipped (no data).")
    return price_data


# -- Step 2b: Daily tech score series --------------------------------------
def compute_tech_score_series(price_data: dict, sim_start: date) -> tuple:
    """
    For every ticker in price_data, computes daily tech_score and MA crossover state
    for each trading day on or after sim_start.

    tech_score  = (price/MA50 - 1) + (MA50/MA200 - 1)
    ma_crossover = True when price >= MA50 (local uptrend confirmed)
                   False when price <  MA50 (price below short-term average)

    Note: MA200 is no longer used as a gate. The golden-cross requirement
    (MA50 >= MA200) was removed after backtesting showed it delayed entries
    by months post-crash, causing the strategy to buy at local highs after
    the recovery had already happened. MA50 alone responds weeks earlier.

    Returns: (tech_score_series, ma_crossover_series)
        tech_score_series   : dict[ticker -> dict[date -> float]]
        ma_crossover_series : dict[ticker -> dict[date -> bool]]
    """
    tech_series  = {}
    cross_series = {}
    for ticker, df in price_data.items():
        closes = df['Close']
        ma50   = closes.rolling(50, min_periods=50).mean()
        ma200  = closes.rolling(200, min_periods=200).mean()

        # Both MAs must be valid; price_gap and ma_gap match data.py exactly
        price_gap = (closes - ma50)  / ma50
        ma_gap    = (ma50   - ma200) / ma200
        tech      = price_gap + ma_gap

        ticker_scores:    dict = {}
        ticker_crossover: dict = {}
        for idx in tech.index:
            d   = idx.date() if hasattr(idx, 'date') else idx
            val = tech[idx]
            m50 = ma50[idx]
            m200= ma200[idx]
            if d >= sim_start and not pd.isna(val):
                ticker_scores[d]    = float(val)
                ticker_crossover[d] = (not pd.isna(m50)
                                       and float(closes[idx]) >= float(m50))
        tech_series[ticker]  = ticker_scores
        cross_series[ticker] = ticker_crossover

    covered = sum(1 for v in tech_series.values() if v)
    print(f"  Daily tech scores computed for {covered}/{len(price_data)} tickers "
          f"(from {sim_start}).")
    return tech_series, cross_series


# -- Step 2c: Daily RSI series ----------------------------------------------
def compute_rsi_series(price_data: dict, sim_start: date, period: int = 14) -> dict:
    """
    Computes daily RSI(14) for every ticker using Wilder's smoothing method.

    RSI = 100 - 100 / (1 + RS)   where RS = avg_gain / avg_loss over `period` days.

    Used as Gate 6 in passes_c3(): only enter when RSI is in [RSI_ENTRY_MIN, RSI_ENTRY_MAX].
    This ensures we buy on a healthy pullback within an uptrend rather than chasing
    a stock that is already locally overbought.

    Returns: dict[ticker -> dict[date -> float]]  (only dates >= sim_start)
    """
    rsi_series = {}
    for ticker, df in price_data.items():
        closes = df['Close']
        delta  = closes.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder smoothing (equivalent to EWM with alpha = 1/period)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))

        ticker_rsi: dict = {}
        for idx, val in rsi.items():
            d = idx.date() if hasattr(idx, 'date') else idx
            if d >= sim_start and not pd.isna(val):
                ticker_rsi[d] = float(val)
        rsi_series[ticker] = ticker_rsi

    covered = sum(1 for v in rsi_series.values() if v)
    print(f"  Daily RSI({period}) computed for {covered}/{len(price_data)} tickers "
          f"(from {sim_start}).")
    return rsi_series


# -- Step 2d: Daily ATR series ---------------------------------------------
def compute_atr_series(price_data: dict, sim_start: date, period: int = 14) -> dict:
    """
    Computes daily ATR(14) for every ticker using Wilder's exponential smoothing.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = EWM of TR with alpha = 1/period (Wilder smoothing)

    Used to set adaptive trailing stop distances:
        stop_dist = ATR(14) * ATR_MULTIPLIER   (default: 2× ATR)

    This replaces the flat STOP_LOSS_PCT stop, giving each stock a stop width
    proportional to its own typical daily range instead of a one-size-fits-all 5%.

    Returns: dict[ticker -> dict[date -> float]]  (only dates >= sim_start)
    """
    atr_series = {}
    for ticker, df in price_data.items():
        high   = df['High'].squeeze()
        low    = df['Low'].squeeze()
        close  = df['Close'].squeeze()
        prev_c = close.shift(1)
        tr = pd.concat([
            (high - low).rename('hl'),
            (high - prev_c).abs().rename('hc'),
            (low  - prev_c).abs().rename('lc'),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        ticker_atr: dict = {}
        for idx, val in atr.items():
            d = idx.date() if hasattr(idx, 'date') else idx
            if d >= sim_start and not pd.isna(val):
                ticker_atr[d] = float(val)
        atr_series[ticker] = ticker_atr

    covered = sum(1 for v in atr_series.values() if v)
    print(f"  Daily ATR({period}) computed for {covered}/{len(price_data)} tickers "
          f"(from {sim_start}).")
    return atr_series


# -- Step 3: Sentiment cache pre-build -------------------------------------
def build_sentiment_caches(
    ranked_candidates: list,
    trading_dates: list,
    sentiment_agent,
    news_fetcher,
    skip: bool = False,
    max_cache_tickers: int = 0,
):
    """
    Pre-builds sentiment_cache_{TICKER}.csv for every candidate over the backtest window.
    Loads and returns cache_data: dict of {ticker: DataFrame(score, consensus_pct, headline_count, momentum)}.

    max_cache_tickers: if > 0, only build caches for the top N ranked candidates.
                       Remaining tickers use the tech-signal no-cache fallback.
                       Useful to limit build time when there are many candidates.
    """
    cache_data = {}
    build_list = ranked_candidates if max_cache_tickers <= 0 else ranked_candidates[:max_cache_tickers]

    print(f"\n{'='*65}")
    if skip:
        print(f"  [Step 2/3] Loading existing sentiment caches (--skip-cache-build)...")
    else:
        n = len(build_list)
        est_min = n * len(trading_dates) * 0.1 / 60
        print(f"  [Step 2/3] Pre-building sentiment caches for {n} candidates (~{est_min:.0f} min)...")
        print(f"  (This is a one-time step. Re-runs load from cache instantly.)")
    print(f"{'='*65}")

    for i, candidate in enumerate(ranked_candidates):
        ticker = candidate['ticker']
        if not skip and candidate in build_list:
            try:
                build_or_update_cache(
                    ticker, trading_dates, sentiment_agent, news_fetcher, _DEFAULT_CACHE_DIR,
                    sleep_sec=0.1,
                )
            except Exception as e:
                print(f"  [CACHE] {ticker}: cache build failed ({e}). Will use neutral fallback.")
        df = _load_cache(ticker, _DEFAULT_CACHE_DIR)
        cache_data[ticker] = df
        if skip and (i + 1) % 20 == 0:
            print(f"  Loaded {i+1}/{len(ranked_candidates)} caches...")

    cached_count = sum(1 for df in cache_data.values() if not df.empty)
    print(f"\n  {cached_count}/{len(ranked_candidates)} tickers have sentiment data.")
    return cache_data


# -- C3 sentiment gate ------------------------------------------------------
def passes_c3(candidate: dict, current_date: date, cache_data: dict,
              tech_score_series: dict = None,
              ma_crossover_series: dict = None,
              rsi_series: dict = None) -> tuple:
    """
    C3 gate — Option A: each criterion is a hard independent floor.

    Gate 0 (MA crossover): MA50 >= MA200 — golden cross required.
                           If MA50 < MA200 the stock is in a confirmed downtrend;
                           no other signal can override this block.

    Gate 1 (tech):      tech_signal_norm >= TECH_FLOOR (0.3)
                        Stock must not be in a downtrend (price/MA50/MA200 all checked).
                        This is a binary block, not a compensating score.

    Gate 2 (sentiment): LLM score >= SENTIMENT_FLOOR (6.5 / 10)
                        The LLM must be genuinely positive. Cannot be overridden by
                        a strong tech signal.

    Gate 3 (consensus): >= 70% of articles agree with the sentiment direction.

    Gate 4 (momentum):  Sentiment is flat or improving day-over-day (>= 0).

    Gate 5 (headlines): >= MIN_HEADLINES articles available for a meaningful reading.

    Gate 6 (RSI):       RSI(14) >= RSI_ENTRY_MIN (default 35).
                        Floor only — avoids buying into a falling knife.
                        Ceiling removed: sustained momentum runs (RSI > 55) are valid entries.
                        Above 55 → don't chase overbought rallies.

    No real sentiment available (no cache / < MIN_HEADLINES):
        Skip entirely -- do NOT fall back to tech-only. A missing sentiment reading
        is treated as insufficient evidence, not neutral evidence.

    Returns (passes: bool, score: float, reason: str).
    The score returned is the raw LLM sentiment score (0-10) for logging/display.
    """
    ticker = candidate['ticker']
    df     = cache_data.get(ticker)

    # Gate 0: price must be above MA50 (local uptrend confirmed).
    # MA200 requirement removed — the golden-cross gate delayed entries by months
    # after a crash recovery, causing buys at local highs. MA50 alone fires weeks earlier.
    if ma_crossover_series and ticker in ma_crossover_series:
        above_ma50 = ma_crossover_series[ticker].get(current_date, False)
        if not above_ma50:
            return False, 0.0, "price<MA50(downtrend)"

    # Live tech score: look up today's computed value; fall back to pipeline snapshot
    if tech_score_series and ticker in tech_score_series:
        raw_tech = tech_score_series[ticker].get(current_date,
                                                  candidate.get('tech_score', 0.0))
    else:
        raw_tech = candidate.get('tech_score', 0.0)

    tech_signal_norm = min(max(raw_tech, 0.0) / 0.15, 1.0)

    # Gate 1: tech floor — must not be in a downtrend
    if tech_signal_norm < TECH_FLOOR:
        return False, 0.0, f"tech_norm={tech_signal_norm:.2f}<{TECH_FLOOR}(downtrend)"

    # Gate 6: RSI pullback filter — only enter on healthy dips within an uptrend.
    # Skipped if rsi_series is not provided (preserves backward compatibility).
    if rsi_series and ticker in rsi_series:
        rsi = rsi_series[ticker].get(current_date)
        if rsi is not None:
            if rsi < RSI_ENTRY_MIN:
                return False, 0.0, f"RSI={rsi:.1f}<{RSI_ENTRY_MIN}(oversold/broken)"

    # Gates 2-5 require real sentiment data
    if df is None or df.empty or current_date not in df.index:
        return False, 0.0, "no_sentiment_cache"

    row       = df.loc[current_date]
    headlines = int(row.get('headline_count', 0))

    # Gate 5: enough headlines for a meaningful reading
    if headlines < MIN_HEADLINES:
        return False, 0.0, f"headlines={headlines}<{MIN_HEADLINES}"

    sentiment_score = float(row['score'])
    consensus       = float(row['consensus_pct'])
    momentum        = int(row.get('momentum', 0))

    # Gate 2: sentiment floor — LLM must be genuinely positive
    if sentiment_score < SENTIMENT_FLOOR:
        return False, sentiment_score, f"sentiment={sentiment_score:.1f}<{SENTIMENT_FLOOR}"

    # Gate 3: consensus
    if consensus < CONSENSUS_THRESHOLD:
        return False, sentiment_score, f"consensus={consensus:.0%}<70%"

    # Gate 4: momentum — not worsening
    if momentum < 0:
        return False, sentiment_score, f"momentum={momentum}(worsening)"

    return True, sentiment_score, f"sentiment={sentiment_score:.1f},consensus={consensus:.0%},tech={tech_signal_norm:.2f},n={headlines}"


def find_next_buy(
    ranked_candidates: list,
    open_tickers: set,
    current_date: date,
    cache_data: dict,
    price_data: dict,
    tech_score_series: dict = None,
    ma_crossover_series: dict = None,
    rsi_series: dict = None,
    last_exit_dates: dict = None,
) -> tuple:
    """
    Scans ranked candidates in order, applying C3 sentiment gate.
    Returns (candidate dict, final_score) for the first passing candidate, or (None, 0).
    """
    for candidate in ranked_candidates:
        ticker = candidate['ticker']
        if ticker in open_tickers:
            continue
        if last_exit_dates and ticker in last_exit_dates:
            days_since_exit = (current_date - last_exit_dates[ticker]).days
            if days_since_exit < REENTRY_COOLDOWN_DAYS:
                continue
        if ticker not in price_data:
            continue
        ok, score, reason = passes_c3(candidate, current_date, cache_data,
                                       tech_score_series, ma_crossover_series, rsi_series)
        if ok:
            return candidate, score
    return None, 0.0


# -- Portfolio simulation ---------------------------------------------------
def run_simulation(
    ranked_candidates: list,
    price_data: dict,
    cache_data: dict,
    backtest_start: str,
    backtest_end: str,
    initial_capital: float = INITIAL_CAPITAL,
    tech_score_series: dict = None,
    ma_crossover_series: dict = None,
    rsi_series: dict = None,
    atr_series: dict = None,
) -> dict:
    """
    Day-by-day portfolio simulation.
    Returns results dict with equity series, trades, metrics.
    """
    # Use SPY's date index as the canonical trading calendar
    if 'SPY' not in price_data:
        print("  ERROR: SPY price data missing. Cannot run simulation.")
        sys.exit(1)

    spy_prices_full = price_data['SPY']['Close']
    sim_start_date  = date.fromisoformat(backtest_start)
    # Restrict simulation to dates on or after backtest_start
    # (price_data may include earlier rows for MA warm-up)
    spy_prices    = spy_prices_full[spy_prices_full.index >= pd.Timestamp(backtest_start)]
    trading_dates = [d.date() if hasattr(d, 'date') else d for d in spy_prices.index]

    # SPY regime filter: compute MA50 over full extended history (includes warm-up)
    # so Day 1 already has a valid MA. Used daily to cap effective max positions:
    #   SPY >= MA50  → bull regime  → MAX_OPEN_POSITIONS slots available
    #   SPY <  MA50  → bear regime  → max 2 slots (defensive, don't add new risk)
    spy_ma50_full = spy_prices_full.rolling(50).mean()
    spy_ma50 = {
        (idx.date() if hasattr(idx, 'date') else idx): float(v)
        for idx, v in spy_ma50_full.items()
        if not pd.isna(v)
    }
    SPY_BEAR_MAX_POSITIONS = 2

    print(f"\n{'='*65}")
    print(f"  [Step 3/3] Portfolio simulation ({backtest_start} -> {backtest_end})")
    print(f"{'='*65}")

    cash            = initial_capital
    open_positions: list[Position]     = []
    closed_trades:  list[ClosedTrade]  = []
    portfolio_values = []
    last_exit_dates: dict = {}   # ticker -> date of last exit
    spy_values       = []
    spy_start        = float(spy_prices.iloc[0])

    # Pending entries: (candidate, entry_date) -- entered at NEXT day's open
    pending_entries = []

    # Track which rank indices have been used (for replacement logic)
    used_rank_indices = set()

    # Pre-compute MA50 lookup for trend-failure exit (close < MA50 → exit position)
    ma50_series: dict = {}
    for ticker, df in price_data.items():
        closes = df['Close']
        ma50   = closes.rolling(50, min_periods=50).mean()
        ticker_ma50: dict = {}
        for idx, val in ma50.items():
            d = idx.date() if hasattr(idx, 'date') else idx
            if d >= sim_start_date and not pd.isna(val):
                ticker_ma50[d] = float(val)
        ma50_series[ticker] = ticker_ma50

    # -- Helper: get price row for ticker/date ------------------------------
    def get_price_row(ticker: str, d: date):
        df = price_data.get(ticker)
        if df is None:
            return None
        idx = [x.date() if hasattr(x, 'date') else x for x in df.index]
        if d not in idx:
            return None
        return df.iloc[idx.index(d)]

    def get_close(ticker: str, d: date, fallback=None):
        row = get_price_row(ticker, d)
        return float(row['Close']) if row is not None else fallback

    # -- Day 1: Open initial positions (regime-aware) ----------------------
    first_date = trading_dates[0]
    spy_d1     = float(spy_prices.iloc[0])
    spy_ma50_d1 = spy_ma50.get(first_date)
    d1_bear     = spy_ma50_d1 is not None and spy_d1 < spy_ma50_d1
    d1_max      = SPY_BEAR_MAX_POSITIONS if d1_bear else MAX_OPEN_POSITIONS
    print(f"\n  Day 1 ({first_date}): Opening initial positions "
          f"[SPY regime: {'BEAR — max %d slots' % SPY_BEAR_MAX_POSITIONS if d1_bear else 'BULL — max %d slots' % MAX_OPEN_POSITIONS}]...")
    initial_opened = []
    for i, candidate in enumerate(ranked_candidates):
        if len(initial_opened) >= d1_max:
            break
        ticker = candidate['ticker']
        if ticker not in price_data:
            continue
        ok, score, reason = passes_c3(candidate, first_date, cache_data,
                                       tech_score_series, ma_crossover_series, rsi_series)
        if not ok:
            print(f"    {ticker}: C3 FAIL ({reason}) -- skipping initial entry")
            continue
        row = get_price_row(ticker, first_date)
        if row is None:
            continue
        entry_price = float(row['Open'])

        # Risk-based sizing: risk 2% of capital per trade, stop width = ATR×2
        atr_val      = (atr_series or {}).get(ticker, {}).get(first_date,
                        entry_price * STOP_LOSS_PCT)
        stop_dist    = atr_val * ATR_MULTIPLIER
        risk_dollars = initial_capital * RISK_PER_TRADE_PCT
        shares       = max(1, int(risk_dollars / stop_dist))
        # Cap: single position at most MAX_POSITION_PCT of cash
        max_by_cap   = max(1, int(cash * MAX_POSITION_PCT / entry_price))
        shares       = min(shares, max_by_cap)
        cost = shares * entry_price
        if cost > cash:
            continue
        cash -= cost
        pos = Position(
            ticker=ticker, entry_date=first_date, entry_price=entry_price,
            shares=shares,
            stop_price=round(entry_price - stop_dist, 2),
            target_price=round(entry_price * (1 + TAKE_PROFIT_PCT), 2),
            rank_index=i,
            peak_price=entry_price,
            atr_stop_dist=stop_dist,
        )
        open_positions.append(pos)
        used_rank_indices.add(i)
        initial_opened.append(ticker)
        print(f"    {ticker}: {shares} shares @ ${entry_price:.2f} "
              f"(stop=${pos.stop_price:.2f}, target=${pos.target_price:.2f}, C3={score:.3f})")

    print(f"\n  Initial portfolio: {[p.ticker for p in open_positions]}")
    if len(open_positions) < MAX_OPEN_POSITIONS:
        print(f"  WARNING: Only {len(open_positions)}/{MAX_OPEN_POSITIONS} initial positions "
              f"(some candidates failed C3 or lacked data)")

    # -- Main simulation loop -----------------------------------------------
    replacements_today = []   # entries to open next day

    for day_i, current_date in enumerate(trading_dates):
        # -- A: Execute pending entries from yesterday ----------------------
        for (cand, _entry_date) in pending_entries:
            ticker = cand['ticker']
            row    = get_price_row(ticker, current_date)
            if row is None:
                continue
            entry_price = float(row['Open'])
            # Risk-based sizing: risk 2% of current portfolio value per trade, stop = ATR×2
            port_val     = cash + sum(
                (get_close(p.ticker, current_date) or p.entry_price) * p.shares
                for p in open_positions
            )
            atr_val      = (atr_series or {}).get(ticker, {}).get(current_date,
                            entry_price * STOP_LOSS_PCT)
            stop_dist    = atr_val * ATR_MULTIPLIER
            risk_dollars = port_val * RISK_PER_TRADE_PCT
            shares       = max(1, int(risk_dollars / stop_dist))
            max_by_cap   = max(1, int(cash * MAX_POSITION_PCT / entry_price))
            shares       = min(shares, max_by_cap)
            if shares < 1 or shares * entry_price > cash:
                print(f"  {current_date}: {ticker} -- insufficient cash (${cash:.0f}). Holding cash.")
                continue
            cash -= shares * entry_price
            rank_idx = ranked_candidates.index(cand) if cand in ranked_candidates else -1
            pos = Position(
                ticker=ticker, entry_date=current_date, entry_price=entry_price,
                shares=shares,
                stop_price=round(entry_price - stop_dist, 2),
                target_price=round(entry_price * (1 + TAKE_PROFIT_PCT), 2),
                rank_index=rank_idx,
                peak_price=entry_price,
                atr_stop_dist=stop_dist,
            )
            open_positions.append(pos)
            used_rank_indices.add(rank_idx)
            print(f"  {current_date}: ENTER {ticker} {shares}sh @ ${entry_price:.2f} "
                  f"(stop=${pos.stop_price:.2f}, target=${pos.target_price:.2f})")

        pending_entries = []

        # -- B: Check exits (trailing stop updated daily) -------------------
        closed_today = []
        for pos in open_positions[:]:
            row = get_price_row(pos.ticker, current_date)
            if row is None:
                continue  # no data -- forward-hold
            low, high, open_px = float(row['Low']), float(row['High']), float(row['Open'])

            # Advance the trailing stop: if today's high exceeds the peak,
            # raise the stop by the same fixed ATR distance (set at entry).
            if high > pos.peak_price:
                pos.peak_price = high
                pos.stop_price = round(pos.peak_price - pos.atr_stop_dist, 2)

            exit_price  = None
            exit_reason = None

            if low <= pos.stop_price:
                # Gap-down handling: fill at min(open, stop)
                exit_price  = min(open_px, pos.stop_price)
                exit_reason = 'STOP_LOSS'
            elif high >= pos.target_price:
                exit_price  = pos.target_price
                exit_reason = 'TAKE_PROFIT'
            else:
                # Trend failure exit: price closed below MA50 — uptrend has broken.
                # Only activates after TREND_FAIL_MIN_HOLD trading days to avoid
                # conflicting with RSI-dip entries (which buy near/below MA50).
                close_px   = float(row['Close'])
                days_held  = sum(
                    1 for d in trading_dates[:day_i + 1] if d >= pos.entry_date
                )
                if (days_held >= TREND_FAIL_MIN_HOLD
                        and ma50_series and pos.ticker in ma50_series):
                    ma50_today = ma50_series[pos.ticker].get(current_date)
                    if ma50_today is not None and close_px < ma50_today:
                        exit_price  = open_px
                        exit_reason = 'TREND_FAIL'

                # Sentiment sell signal: exit at today's open if LLM turns
                # decisively negative (<SENTIMENT_SELL_FLOOR) with high conviction.
                if not exit_reason:
                    sent_df = cache_data.get(pos.ticker)
                    if sent_df is not None and not sent_df.empty and current_date in sent_df.index:
                        sent_row   = sent_df.loc[current_date]
                        sent_score = float(sent_row.get('score', 10.0))
                        sent_conv  = float(sent_row.get('consensus_pct', 0.0))
                        sent_hl    = int(sent_row.get('headline_count', 0))
                        if (sent_score < SENTIMENT_SELL_FLOOR
                                and sent_conv >= SENTIMENT_SELL_CONV
                                and sent_hl >= MIN_HEADLINES):
                            exit_price  = open_px
                            exit_reason = 'SENTIMENT_SELL'

            if exit_reason:
                cash += pos.shares * exit_price
                trade = ClosedTrade(
                    ticker=pos.ticker, entry_date=pos.entry_date,
                    entry_price=pos.entry_price, shares=pos.shares,
                    exit_date=current_date, exit_price=exit_price,
                    exit_reason=exit_reason, rank_at_entry=pos.rank_index,
                )
                closed_trades.append(trade)
                closed_today.append(pos)
                last_exit_dates[pos.ticker] = current_date
                pnl = (exit_price - pos.entry_price) / pos.entry_price * 100
                print(f"  {current_date}: EXIT {pos.ticker} @ ${exit_price:.2f} "
                      f"({exit_reason}, {'+' if pnl>=0 else ''}{pnl:.1f}%)")

        for pos in closed_today:
            open_positions.remove(pos)

        # -- C: Fill all open portfolio slots (replacements + proactive fill) -
        # Runs whenever open positions + pending entries < effective_max_positions.
        # SPY regime filter: cap new entries at 2 when SPY is below its MA50.
        spy_close_today = float(spy_prices.get(pd.Timestamp(current_date),
                                               spy_prices.iloc[-1]))
        spy_ma50_today  = spy_ma50.get(current_date)
        in_bear_regime  = spy_ma50_today is not None and spy_close_today < spy_ma50_today
        effective_max   = SPY_BEAR_MAX_POSITIONS if in_bear_regime else MAX_OPEN_POSITIONS

        pending_tickers = {c['ticker'] for c, _ in pending_entries}
        all_occupied    = {p.ticker for p in open_positions} | pending_tickers
        reported_no_entry = False
        while len(open_positions) + len(pending_entries) < effective_max:
            next_cand, score = find_next_buy(
                ranked_candidates, all_occupied, current_date, cache_data, price_data,
                tech_score_series, ma_crossover_series, rsi_series,
                last_exit_dates
            )
            if next_cand is None:
                # Only report once per day, and only when we expected a replacement
                if closed_today and not reported_no_entry:
                    print(f"  {current_date}: No qualifying entry found -- holding cash")
                    reported_no_entry = True
                break
            label = 'REPLACEMENT' if closed_today else 'ENTRY'
            print(f"  {current_date}: {label} -> {next_cand['ticker']} "
                  f"(C3 score={score:.3f}) -- entering tomorrow")
            pending_entries.append((next_cand, current_date))
            all_occupied.add(next_cand['ticker'])

        # -- D: Mark-to-market ----------------------------------------------
        position_value = 0.0
        for pos in open_positions:
            close = get_close(pos.ticker, current_date)
            if close is None:
                close = pos.entry_price  # fallback: hold at cost
            position_value += pos.shares * close

        port_value = cash + position_value
        portfolio_values.append((current_date, port_value))
        spy_close = float(spy_prices.iloc[day_i])
        spy_values.append((current_date, initial_capital * spy_close / spy_start))

    # -- Mark remaining open positions as "OPEN" in the trade log ----------
    last_date = trading_dates[-1]
    for pos in open_positions:
        last_close = get_close(pos.ticker, last_date) or pos.entry_price
        closed_trades.append(ClosedTrade(
            ticker=pos.ticker, entry_date=pos.entry_date,
            entry_price=pos.entry_price, shares=pos.shares,
            exit_date=last_date, exit_price=last_close,
            exit_reason='OPEN', rank_at_entry=pos.rank_index,
        ))

    return {
        'portfolio_values': portfolio_values,
        'spy_values':       spy_values,
        'closed_trades':    closed_trades,
        'initial_capital':  initial_capital,
        'ranked_candidates': ranked_candidates,
    }


# -- Metrics ----------------------------------------------------------------
def compute_metrics(results: dict) -> dict:
    pv  = pd.Series({d: v for d, v in results['portfolio_values']})
    spv = pd.Series({d: v for d, v in results['spy_values']})
    ic  = results['initial_capital']

    daily_ret = pv.pct_change().dropna()
    n_days    = len(daily_ret)
    ann_ret   = (pv.iloc[-1] / ic) ** (252 / max(n_days, 1)) - 1
    ann_vol   = daily_ret.std() * (252 ** 0.5) if n_days > 1 else 0.0
    sharpe    = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0
    max_dd    = ((pv / pv.cummax()) - 1).min()

    spy_ret = (spv.iloc[-1] / ic) - 1
    alpha   = (pv.iloc[-1] / ic - 1) - spy_ret

    trades    = results['closed_trades']
    stops     = sum(1 for t in trades if t.exit_reason == 'STOP_LOSS')
    targets   = sum(1 for t in trades if t.exit_reason == 'TAKE_PROFIT')
    still_open = sum(1 for t in trades if t.exit_reason == 'OPEN')
    winners   = sum(1 for t in trades if t.exit_price > t.entry_price)

    return {
        'ann_return': ann_ret,
        'total_return': (pv.iloc[-1] / ic) - 1,
        'spy_return':  spy_ret,
        'alpha':       alpha,
        'sharpe':      sharpe,
        'max_drawdown': max_dd,
        'ann_vol':     ann_vol,
        'n_trades':    len(trades),
        'n_stops':     stops,
        'n_targets':   targets,
        'n_open':      still_open,
        'n_winners':   winners,
        'end_value':   pv.iloc[-1],
    }


# -- Output -----------------------------------------------------------------
def save_trades_csv(results: dict, output_dir: str):
    trades = results['closed_trades']
    rows = []
    for t in trades:
        pnl_pct = (t.exit_price - t.entry_price) / t.entry_price * 100
        days    = (t.exit_date - t.entry_date).days
        rows.append({
            'ticker':        t.ticker,
            'entry_date':    t.entry_date,
            'entry_price':   round(t.entry_price, 2),
            'shares':        t.shares,
            'exit_date':     t.exit_date,
            'exit_price':    round(t.exit_price, 2),
            'exit_reason':   t.exit_reason,
            'pnl_pct':       round(pnl_pct, 2),
            'holding_days':  days,
            'rank_at_entry': t.rank_at_entry,
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'portfolio_trades.csv')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  Trade log saved: {path}")
    return path


def plot_equity_curve(results: dict, metrics: dict, backtest_start: str,
                      backtest_end: str, output_dir: str):
    pv  = pd.Series({d: v for d, v in results['portfolio_values']})
    spv = pd.Series({d: v for d, v in results['spy_values']})
    ic  = results['initial_capital']

    pv.index  = pd.to_datetime(pv.index)
    spv.index = pd.to_datetime(spv.index)

    strat_pct = (pv / ic - 1) * 100
    spy_pct   = (spv / ic - 1) * 100

    # Replacement event dates
    trades     = results['closed_trades']
    repl_dates = pd.to_datetime([t.exit_date for t in trades
                                 if t.exit_reason in ('STOP_LOSS', 'TAKE_PROFIT')])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # -- Top panel: portfolio value -----------------------------------------
    ax1.plot(pv.index, pv.values, color='steelblue', linewidth=1.8,
             label='Portfolio')
    ax1.plot(spv.index, spv.values, color='grey', linewidth=1.2,
             linestyle='--', label='SPY B&H', alpha=0.7)
    for rd in repl_dates:
        ax1.axvline(x=rd, color='orange', alpha=0.4, linewidth=0.8)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'Portfolio Backtest  |  {backtest_start} -> {backtest_end}  '
                  f'|  S&P 500 5-Criteria Strategy')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3)

    # Metrics annotation
    txt = (f"Return: {metrics['total_return']*100:+.1f}%  |  "
           f"SPY: {metrics['spy_return']*100:+.1f}%  |  "
           f"Alpha: {metrics['alpha']*100:+.1f}%\n"
           f"Sharpe: {metrics['sharpe']:.2f}  |  "
           f"Max DD: {metrics['max_drawdown']*100:.1f}%  |  "
           f"Trades: {metrics['n_trades']}  "
           f"(stop:{metrics['n_stops']} tp:{metrics['n_targets']} open:{metrics['n_open']})")
    ax1.text(0.02, 0.03, txt, transform=ax1.transAxes,
             fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # -- Bottom panel: cumulative return % ---------------------------------
    ax2.plot(strat_pct.index, strat_pct.values, color='steelblue', linewidth=1.5,
             label='Strategy')
    ax2.plot(spy_pct.index, spy_pct.values, color='grey', linewidth=1.2,
             linestyle='--', label='SPY', alpha=0.7)
    ax2.fill_between(strat_pct.index, strat_pct.values, 0,
                     where=(strat_pct.values >= 0), color='steelblue', alpha=0.15)
    ax2.fill_between(strat_pct.index, strat_pct.values, 0,
                     where=(strat_pct.values < 0), color='red', alpha=0.15)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'portfolio_equity_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Equity curve saved: {path}")
    return path


def print_summary(results: dict, metrics: dict, backtest_start: str, backtest_end: str):
    rc  = results['ranked_candidates']
    ic  = results['initial_capital']
    m   = metrics

    print(f"\n{'='*65}")
    print(f"  PORTFOLIO BACKTEST SUMMARY  ({backtest_start} -> {backtest_end})")
    print(f"{'='*65}")
    print(f"  Candidates from pipeline : {len(rc)}")
    print(f"  Starting capital         : ${ic:,.0f}")
    print(f"  Ending capital           : ${m['end_value']:,.2f}")
    print(f"\n  RETURNS")
    print(f"    Strategy total return  : {m['total_return']*100:+.2f}%")
    print(f"    SPY buy & hold         : {m['spy_return']*100:+.2f}%")
    print(f"    Alpha vs SPY           : {m['alpha']*100:+.2f}%")
    print(f"\n  RISK METRICS")
    print(f"    Sharpe ratio           : {m['sharpe']:.2f}")
    print(f"    Max drawdown           : {m['max_drawdown']*100:.1f}%")
    print(f"    Annualised volatility  : {m['ann_vol']*100:.1f}%")
    print(f"\n  TRADES  ({m['n_trades']} total)")
    print(f"    Stop-loss exits        : {m['n_stops']}")
    print(f"    Take-profit exits      : {m['n_targets']}")
    print(f"    Still open at end      : {m['n_open']}")
    if m['n_trades'] > 0:
        print(f"    Winning trades         : {m['n_winners']}/{m['n_trades']} "
              f"({m['n_winners']/m['n_trades']*100:.0f}%)")
    print(f"\n  NOTE: S&P 500 constituent list from Wikipedia (live fetch).")
    print(f"        Survivorship bias is an acknowledged limitation.")
    print(f"{'='*65}")


# -- Entry point ------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Portfolio backtest: S&P 500 5-criteria strategy with dynamic replacement'
    )
    parser.add_argument('--backtest-start', type=str,
                        default=(date.today() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Backtest start date YYYY-MM-DD (default: 1 year ago)')
    parser.add_argument('--capital',    type=float, default=INITIAL_CAPITAL,
                        help=f'Starting capital (default: {INITIAL_CAPITAL:,.0f})')
    parser.add_argument('--inflation',  type=float, default=0.035,
                        help='C1 inflation rate (default: 3.5%%)')
    parser.add_argument('--skip-cache-build', action='store_true',
                        help='Skip pre-building sentiment caches (use existing CSVs only)')
    parser.add_argument('--max-cache-tickers', type=int, default=40,
                        help='Build sentiment caches for top N ranked tickers only (default: 40). '
                             'Set 0 to build for all candidates.')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated ticker override list (e.g. AAPL,MSFT,NVDA). '
                             'Skips S&P 500 universe fetch; runs full pipeline on these tickers only. '
                             'Best used with high-news-volume stocks to exercise the sentiment layer.')
    args = parser.parse_args()

    backtest_start = args.backtest_start
    backtest_end   = date.today().strftime('%Y-%m-%d')

    # Parse optional ticker override
    ticker_override = [t.strip().upper() for t in args.tickers.split(',')] if args.tickers else None

    # -- Stage 1: Historical pipeline ---------------------------------------
    ranked_candidates, sectors = run_historical_pipeline(
        backtest_start, inflation_rate=args.inflation,
        ticker_override=ticker_override,
    )

    # -- Stage 2: Prefetch price data (includes MA warm-up history) ---------
    candidate_tickers = [c['ticker'] for c in ranked_candidates]
    price_data = prefetch_prices(candidate_tickers, backtest_start, backtest_end)

    # Filter ranked candidates to those with price data
    ranked_candidates = [c for c in ranked_candidates if c['ticker'] in price_data]
    print(f"  {len(ranked_candidates)} candidates with price data for simulation.")

    # Derive canonical trading dates from SPY (simulation window only)
    sim_start = date.fromisoformat(backtest_start)
    spy_dates = [d.date() if hasattr(d, 'date') else d
                 for d in price_data['SPY'].index
                 if (d.date() if hasattr(d, 'date') else d) >= sim_start]

    # Compute daily tech scores, MA crossover states, RSI, and ATR from extended price history
    tech_score_series, ma_crossover_series = compute_tech_score_series(price_data, sim_start)
    rsi_series = compute_rsi_series(price_data, sim_start)
    atr_series = compute_atr_series(price_data, sim_start)

    # -- Stage 3: Sentiment caches ------------------------------------------
    os.environ.setdefault('APCA_API_KEY_ID',     os.getenv('APCA_API_KEY_ID', ''))
    os.environ.setdefault('APCA_API_SECRET_KEY', os.getenv('APCA_API_SECRET_KEY', ''))
    os.environ.setdefault('APCA_API_BASE_URL',   os.getenv('APCA_API_BASE_URL',
                                                            'https://paper-api.alpaca.markets'))

    sentiment_agent = SentimentAgent()
    news_fetcher    = NewsFetcher()

    # When using --tickers override, always build caches for all of them
    max_cache = 0 if ticker_override else args.max_cache_tickers
    cache_data = build_sentiment_caches(
        ranked_candidates, spy_dates, sentiment_agent, news_fetcher,
        skip=args.skip_cache_build,
        max_cache_tickers=max_cache,
    )

    # -- Stage 4: Simulation ------------------------------------------------
    results = run_simulation(
        ranked_candidates, price_data, cache_data,
        backtest_start, backtest_end,
        initial_capital=args.capital,
        tech_score_series=tech_score_series,
        ma_crossover_series=ma_crossover_series,
        rsi_series=rsi_series,
        atr_series=atr_series,
    )

    # -- Stage 5: Metrics + Output ------------------------------------------
    metrics  = compute_metrics(results)
    print_summary(results, metrics, backtest_start, backtest_end)
    save_trades_csv(results, _DEFAULT_OUTPUT_DIR)
    plot_equity_curve(results, metrics, backtest_start, backtest_end, _DEFAULT_OUTPUT_DIR)
