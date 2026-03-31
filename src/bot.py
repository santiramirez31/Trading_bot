"""
bot.py -- LLM-Driven Trading Bot: Full S&P 500 Pipeline

5-Criteria Funnel:
  C1 Quality   : 10yr annualized return > 8.5% (inflation 3.5% + 5pp), Sharpe > 0
  C4 Liquidity : avg 20d volume > 1M shares, price > $5
  C5 Risk      : 90d volatility < 35%, 1y max drawdown > -30%
  [RANK]       : combined C5 risk score + C2 technical trend strength
  C3 Sentiment : final_score = 0.7 * tech_signal + 0.3 * sentiment >= 0.65
                 AND consensus >= 70%, min 5 headlines, momentum >= 0
  [STOP]       : after 10 open positions reached

Risk controls per trade:
  Stop-loss         : 2% below entry (bracket order, managed by Alpaca)
  Take-profit       : 5% above entry (bracket order, managed by Alpaca)
  Sentiment sell    : close if LLM score drops below 4.0
  Trend-fail exit   : close if price falls below MA50
  Position size     : 10% of portfolio / price, capped at MAX_SHARES=20
  Cooldown          : 24h per ticker before re-entry
  Daily trade cap   : max 2 NEW trades per calendar day
  Portfolio cap     : max 10 open positions at any time
  Sector cap        : max 30% of portfolio in any single GICS sector
  Event-risk filter : skip tickers with earnings within 2 days

Live loop:
  - Runs continuously while the market is open
  - Every SCAN_INTERVAL_MIN minutes: monitors open positions then scans for entries
  - While market is closed: sleeps and waits, recycles the ranked list at open
"""

import os
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', message='.*utcnow.*', category=FutureWarning)

from datetime import datetime, timedelta, date
from alpaca_trade_api.rest import REST

from data import MarketDataHandler, NewsFetcher
from sentiment import SentimentAgent
from industry_weights import TICKER_SECTOR_MAP, normalize_sector

# ── constants ────────────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS    = 10
MAX_NEW_TRADES_DAY    = 2
MAX_SECTOR_PCT        = 0.30
STOP_LOSS_PCT         = 0.02
TAKE_PROFIT_PCT       = 0.05
POSITION_SIZE_PCT     = 0.10
MAX_SHARES            = 20
COOLDOWN_HOURS        = 24
FINAL_SCORE_THRESHOLD = 0.65
SENTIMENT_SELL_FLOOR  = 4.0    # close position if LLM score drops below this
MA50_WINDOW           = 50     # days for trend-fail check
SCAN_INTERVAL_MIN     = 15     # minutes between scans during market hours
SLEEP_CLOSED_MIN      = 5      # minutes to sleep when market is closed


class TradingBot:
    def __init__(self, sentiment_agent: SentimentAgent):
        self.api_key    = os.getenv('APCA_API_KEY_ID')
        self.api_secret = os.getenv('APCA_API_SECRET_KEY')
        self.base_url   = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        self.api             = REST(self.api_key, self.api_secret, self.base_url)
        self.news_fetcher    = NewsFetcher()
        self.sentiment_agent = sentiment_agent
        self._last_trade_time: dict = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        try:
            return self.api.get_clock().is_open
        except Exception:
            return False

    def _get_portfolio_value(self) -> float:
        try:
            return float(self.api.get_account().portfolio_value)
        except Exception:
            return 10_000.0

    def _open_position_count(self) -> int:
        try:
            return len(self.api.list_positions())
        except Exception:
            return 0

    def _in_cooldown(self, ticker: str) -> bool:
        last = self._last_trade_time.get(ticker)
        if last and (datetime.now() - last) < timedelta(hours=COOLDOWN_HOURS):
            rem = COOLDOWN_HOURS - int((datetime.now() - last).seconds / 3600)
            print(f"  [GUARDRAIL] {ticker} in cooldown ({rem}h remaining). Skipping.")
            return True
        return False

    def _has_event_risk(self, ticker: str) -> bool:
        """Skip tickers with earnings within 2 days."""
        try:
            import yfinance as yf
            cal = yf.Ticker(ticker).calendar
            if cal is None or (hasattr(cal, 'empty') and cal.empty):
                return False
            if hasattr(cal, 'columns'):
                earnings_cols = [c for c in cal.columns if 'Earnings' in str(c)]
                if not earnings_cols:
                    return False
                next_earnings = cal[earnings_cols[0]].iloc[0]
            else:
                return False
            if hasattr(next_earnings, 'date'):
                next_earnings = next_earnings.date()
            elif not isinstance(next_earnings, type(date.today())):
                return False
            days_until = (next_earnings - date.today()).days
            if 0 <= days_until <= 2:
                print(f"  [GUARDRAIL] {ticker}: earnings in {days_until} day(s). Skipping event risk.")
                return True
        except Exception:
            pass
        return False

    def _sector_exposure_ok(self, new_sector: str, new_trade_value: float,
                             portfolio_value: float, sectors: dict = None) -> bool:
        """Limit any single sector to MAX_SECTOR_PCT of portfolio."""
        if not new_sector or portfolio_value <= 0:
            return True
        try:
            positions = self.api.list_positions()
            sector_total = 0.0
            for pos in positions:
                raw = (sectors or {}).get(pos.symbol, TICKER_SECTOR_MAP.get(pos.symbol, ''))
                if normalize_sector(raw) == new_sector:
                    sector_total += float(pos.market_value)
            pct_after = (sector_total + new_trade_value) / portfolio_value
            if pct_after > MAX_SECTOR_PCT:
                print(f"  [GUARDRAIL] Sector {new_sector}: would reach {pct_after:.0%} (cap {MAX_SECTOR_PCT:.0%}). Skipping.")
                return False
        except Exception:
            pass
        return True

    # ── position monitoring ──────────────────────────────────────────────────

    def monitor_positions(self, md: 'MarketDataHandler'):
        """
        Check each open position for exit conditions not covered by the bracket order:
          1. Sentiment sell  -- LLM score < SENTIMENT_SELL_FLOOR (4.0)
          2. Trend-fail exit -- price falls below MA50
        Bracket-order exits (stop-loss / take-profit) are managed by Alpaca automatically.
        """
        try:
            positions = self.api.list_positions()
        except Exception as e:
            print(f"  [MONITOR] Could not fetch positions: {e}")
            return

        if not positions:
            print("  [MONITOR] No open positions to monitor.")
            return

        print(f"  [MONITOR] Checking {len(positions)} open position(s)...")

        for pos in positions:
            ticker        = pos.symbol
            current_price = float(pos.current_price)
            qty           = pos.qty

            # ── Trend-fail: price below MA50 ──────────────────────────────
            try:
                warm_start = (date.today() - timedelta(days=MA50_WINDOW + 30)).strftime('%Y-%m-%d')
                prices_df  = md.get_historical_data(ticker, start=warm_start)
                if prices_df is not None and len(prices_df) >= MA50_WINDOW:
                    ma50 = prices_df['Close'].rolling(MA50_WINDOW).mean().iloc[-1]
                    if current_price < ma50:
                        print(f"  [TREND FAIL] {ticker}: ${current_price:.2f} < MA50 ${ma50:.2f} -- closing position.")
                        self._close_position(ticker, qty, reason='TREND_FAIL')
                        continue
            except Exception as e:
                print(f"  [MONITOR] MA50 check failed for {ticker}: {e}")

            # ── Sentiment sell ────────────────────────────────────────────
            try:
                news = self.news_fetcher.get_recent_news(ticker, limit=15)
                if news:
                    result  = self.sentiment_agent.analyze_news(news, ticker=ticker)
                    score   = result['score'] if isinstance(result, dict) else float(result)
                    print(f"  [MONITOR] {ticker}: sentiment score {score:.1f} "
                          f"(sell floor {SENTIMENT_SELL_FLOOR})")
                    if score < SENTIMENT_SELL_FLOOR:
                        print(f"  [SENTIMENT SELL] {ticker}: score {score:.1f} below floor -- closing.")
                        self._close_position(ticker, qty, reason='SENTIMENT_SELL')
            except Exception as e:
                print(f"  [MONITOR] Sentiment check failed for {ticker}: {e}")

    def _close_position(self, ticker: str, qty: str, reason: str = 'MANUAL'):
        """Submit a market sell and cancel any open bracket legs."""
        try:
            # Cancel open orders for this ticker first (bracket legs)
            open_orders = self.api.list_orders(status='open', symbols=[ticker])
            for o in open_orders:
                try:
                    self.api.cancel_order(o.id)
                except Exception:
                    pass

            self.api.submit_order(
                symbol=ticker, qty=qty, side='sell',
                type='market', time_in_force='day'
            )
            print(f"  [OK] {ticker} closed ({reason}). Shares: {qty}")
        except Exception as e:
            print(f"  [!!] Failed to close {ticker}: {e}")

    # ── entry evaluation ─────────────────────────────────────────────────────

    def evaluate_ticker(self, candidate: dict, sectors: dict = None) -> dict | None:
        """C3: Run LLM sentiment on a ranked candidate. Returns enriched dict or None."""
        ticker        = candidate['ticker']
        price         = candidate['price']
        ma50          = candidate['ma50']
        ma200         = candidate['ma200']
        tech_score    = candidate.get('tech_score', 0.0)
        tech_signal_norm = min(max(tech_score, 0.0) / 0.15, 1.0)

        sector = (sectors or {}).get(ticker) or TICKER_SECTOR_MAP.get(ticker)

        print(f"\n{'='*65}")
        print(f"  {ticker}  |  ${price:.2f}  |  MA50: ${ma50:.2f}  |  MA200: ${ma200:.2f}")
        print(f"  Sector: {sector or 'unknown'}  |  Tech score: {tech_score:.4f}  |  Tech norm: {tech_signal_norm:.2f}")
        print(f"  Rank score: {candidate.get('rank_score',0):.3f}  |  "
              f"10yr return: {candidate.get('ann_return',0)*100:.1f}%  |  "
              f"Sharpe: {candidate.get('sharpe',0):.2f}")
        print(f"  90d vol: {candidate.get('volatility',0)*100:.1f}%  |  "
              f"1y max DD: {candidate.get('max_drawdown',0)*100:.1f}%")
        print(f"{'='*65}")

        if self._in_cooldown(ticker):
            return None
        if self._has_event_risk(ticker):
            return None

        print(f"  Fetching news for {ticker}...")
        news = self.news_fetcher.get_recent_news(ticker, limit=15)
        if not news:
            print(f"  No news available. Skipping.")
            return None
        print(f"  {len(news)} headlines retrieved.")

        sentiment = self.sentiment_agent.analyze_news(
            news, sector=sector, ticker=ticker, tech_signal_norm=tech_signal_norm
        )

        final_score = sentiment['final_score']
        consensus   = sentiment['consensus_pct']
        momentum    = sentiment['momentum']
        article_cnt = sentiment['article_count']

        print(f"\n  --- Signal Summary: {ticker} ---")
        print(f"  Final score  : {final_score:.3f}  "
              f"{'[PASS]' if final_score >= FINAL_SCORE_THRESHOLD else '[FAIL]'}  "
              f"(threshold {FINAL_SCORE_THRESHOLD})")
        print(f"  Consensus    : {consensus:.0%}  "
              f"{'[PASS]' if consensus >= 0.70 else '[FAIL]'}  (threshold 70%)")
        print(f"  Momentum     : {'[PASS] improving/stable' if momentum >= 0 else '[FAIL] worsening'}")
        print(f"  Articles     : {article_cnt}  "
              f"{'[PASS]' if article_cnt >= 5 else '[FAIL]'}  (min 5)")

        if sentiment['tradeable']:
            print(f"  -> BUY SIGNAL: all conditions met. Conviction: {final_score:.3f}")
            return {**candidate, 'final_score': final_score, 'sector': sector, 'momentum': momentum}
        else:
            print(f"  -> HOLD: conditions not fully met.")
            return None

    def execute_trade(self, ticker: str, current_price: float,
                      sector: str = None, sectors: dict = None):
        """Submit a bracket order (market buy + stop-loss + take-profit)."""
        try:
            open_positions = self.api.list_positions()
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                print(f"  [GUARDRAIL] Portfolio at capacity ({len(open_positions)}/{MAX_OPEN_POSITIONS}).")
                return
        except Exception:
            pass

        try:
            filled_today = self.api.list_orders(
                status='filled', limit=50, after=str(date.today())
            )
            new_buys_today = sum(1 for o in filled_today if o.side == 'buy')
            if new_buys_today >= MAX_NEW_TRADES_DAY:
                print(f"  [GUARDRAIL] Daily trade cap reached ({new_buys_today}/{MAX_NEW_TRADES_DAY}).")
                return
        except Exception:
            pass

        try:
            pos = self.api.get_position(ticker)
            print(f"  Already holding {pos.qty} shares of {ticker}. Skipping.")
            return
        except Exception:
            pass

        portfolio_value = self._get_portfolio_value()
        shares = min(int(portfolio_value * POSITION_SIZE_PCT / current_price), MAX_SHARES)
        if shares < 1:
            print(f"  Insufficient capital for {ticker} at ${current_price:.2f}. Skipping.")
            return

        resolved_sector = sector or TICKER_SECTOR_MAP.get(ticker, '')
        if resolved_sector and not self._sector_exposure_ok(
            normalize_sector(resolved_sector), shares * current_price, portfolio_value, sectors
        ):
            return

        stop_price   = round(current_price * (1.0 - STOP_LOSS_PCT), 2)
        target_price = round(current_price * (1.0 + TAKE_PROFIT_PCT), 2)

        print(f"\n  Executing BUY: {shares} shares of {ticker} @ ~${current_price:.2f}")
        print(f"  Stop-loss:   ${stop_price:.2f}  ({STOP_LOSS_PCT*100:.0f}% below entry)")
        print(f"  Take-profit: ${target_price:.2f}  ({TAKE_PROFIT_PCT*100:.0f}% above entry)")

        try:
            order = self.api.submit_order(
                symbol=ticker, qty=shares, side='buy', type='market',
                time_in_force='day', order_class='bracket',
                stop_loss={'stop_price': stop_price},
                take_profit={'limit_price': target_price},
            )
            print(f"  [OK] Order submitted! ID: {order.id}")
            self._last_trade_time[ticker] = datetime.now()
        except Exception as e:
            print(f"  [!!] Order failed: {e}")


# ── pipeline helpers ──────────────────────────────────────────────────────────

def build_ranked_candidates(md: MarketDataHandler, args) -> tuple[list, dict]:
    """Run C1 + C4/C5 screens and return (ranked_candidates, sectors)."""
    print('\n  Fetching S&P 500 universe...')
    sp500_tickers, sectors = md.get_sp500_universe()
    if not sp500_tickers:
        print('  Failed to fetch S&P 500 universe. Exiting.')
        return [], {}
    print(f'S&P 500 universe: {len(sp500_tickers)} tickers across {len(set(sectors.values()))} sectors.')

    print(f'\nCriterion 1 -- Quality screen: {len(sp500_tickers)} tickers '
          f'| 10yr return > 8.5% AND Sharpe > 0...')
    quality_results = md.quality_screen_10yr(
        sp500_tickers, inflation_rate=args.inflation, min_excess_pct=0.05
    )
    if not quality_results:
        print('\n  No stocks passed quality screen.')
        return [], sectors
    print(f'  Criterion 1 complete: {len(quality_results)} / {len(sp500_tickers)} passed.')

    print(f'\nCriteria 4+5 -- Liquidity & risk screen: {len(quality_results)} tickers...')
    ranked = md.screen_liquidity_risk_trend(
        quality_results,
        max_volatility=args.max_vol,
        max_drawdown=args.max_dd
    )
    if not ranked:
        print('\n  No stocks passed liquidity/risk screen.')
        return [], sectors

    print(f'\n  Top 10 ranked candidates (by risk+trend score):')
    print(f"  {'TICKER':<6} {'RANK':>5} {'TECH':>6} {'RISK':>6} "
          f"{'10YR%':>6} {'SHARPE':>6} {'VOL%':>6} {'DD%':>6}")
    print(f"  {'-'*6} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for c in ranked[:10]:
        print(f"  {c['ticker']:<6} {c['rank_score']:>5.3f} {c['tech_score']:>6.3f} "
              f"{c['risk_score']:>6.3f} {c['ann_return']*100:>6.1f} {c['sharpe']:>6.2f} "
              f"{c['volatility']*100:>6.1f} {c['max_drawdown']*100:>6.1f}")

    return ranked, sectors


def scan_for_entries(bot: TradingBot, ranked_candidates: list, sectors: dict, max_positions: int):
    """Evaluate ranked candidates in order, stop when portfolio reaches max_positions."""
    open_count = bot._open_position_count()
    if open_count >= max_positions:
        print(f"  Portfolio at capacity ({open_count}/{max_positions}). Skipping entry scan.")
        return

    slots = max_positions - open_count
    print(f"  {open_count}/{max_positions} positions open. "
          f"Scanning {len(ranked_candidates)} candidates for up to {slots} new entries...")

    buys = 0
    for candidate in ranked_candidates:
        if bot._open_position_count() >= max_positions:
            break
        if buys >= slots:
            break
        result = bot.evaluate_ticker(candidate, sectors=sectors)
        if result:
            bot.execute_trade(
                result['ticker'], result['price'],
                sector=result.get('sector'), sectors=sectors
            )
            buys += 1
        time.sleep(0.3)

    if buys == 0:
        print("  No new high-conviction entries found this cycle.")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LLM Trading Bot -- continuous live loop')
    parser.add_argument('--max-positions', type=int,  default=10,   help='Max total open positions at any time (default 10)')
    parser.add_argument('--max-vol',     type=float, default=0.35, help='Max 90d volatility (default 35%%)')
    parser.add_argument('--max-dd',      type=float, default=-0.30, help='Max 1y drawdown (default -30%%)')
    parser.add_argument('--inflation',   type=float, default=0.035, help='Inflation benchmark (default 3.5%%)')
    parser.add_argument('--interval',    type=int,   default=SCAN_INTERVAL_MIN,
                        help=f'Minutes between scans (default {SCAN_INTERVAL_MIN})')
    parser.add_argument('--once',        action='store_true',
                        help='Run a single scan then exit (no continuous loop)')
    args = parser.parse_args()

    print('\n' + '='*65)
    print('  LLM-Driven Trading Bot  |  Student Capital Growth Strategy')
    print(f'  Universe: S&P 500 | Five-Criteria Funnel | Max {args.max_positions} positions')
    print('  Objective: inflation-beating returns with disciplined risk')
    if not args.once:
        print(f'  Loop mode: scan every {args.interval} min | Ctrl+C to stop')
    print('='*65)

    # Apply CLI override to module-level constant
    MAX_OPEN_POSITIONS = args.max_positions

    md            = MarketDataHandler()
    shared_agent  = SentimentAgent()
    bot           = TradingBot(sentiment_agent=shared_agent)

    # Build the ranked candidate list once (refreshed at each market open)
    ranked_candidates, sectors = build_ranked_candidates(md, args)
    if not ranked_candidates:
        print('  Pipeline returned no candidates. Exiting.')
        exit(0)

    last_date_refreshed = date.today()

    # ── single-run mode ───────────────────────────────────────────────────────
    if args.once:
        print('\n  [ONCE] Running single scan...')
        scan_for_entries(bot, ranked_candidates, sectors, args.max_positions)
        exit(0)

    # ── continuous loop ───────────────────────────────────────────────────────
    print('\n  [LIVE] Starting continuous monitoring loop. Press Ctrl+C to stop.\n')
    try:
        while True:
            now = datetime.now()
            ts  = now.strftime('%Y-%m-%d %H:%M')

            # Refresh ranked list once per calendar day
            if date.today() != last_date_refreshed:
                print(f'\n  [{ts}] New trading day -- refreshing candidate list...')
                ranked_candidates, sectors = build_ranked_candidates(md, args)
                last_date_refreshed = date.today()

            if not bot.is_market_open():
                print(f'  [{ts}] Market closed. Next check in {SLEEP_CLOSED_MIN} min...')
                time.sleep(SLEEP_CLOSED_MIN * 60)
                continue

            print(f'\n  [{ts}] ── Market scan ──────────────────────────────────')

            # 1. Monitor existing positions (sentiment sell + trend fail)
            bot.monitor_positions(md)

            # 2. Scan for new entries if slots are available
            scan_for_entries(bot, ranked_candidates, sectors, args.max_positions)

            print(f'\n  [{ts}] Scan complete. Next scan in {args.interval} min...')
            time.sleep(args.interval * 60)

    except KeyboardInterrupt:
        print('\n\n  [STOP] Bot stopped by user. Open positions remain in Alpaca paper account.')
        print('  Bracket orders (stop-loss / take-profit) continue to be managed by Alpaca.')
