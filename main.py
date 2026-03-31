"""
main.py -- LLM-Driven Trading Bot: Entry Point

Runs either the live paper-trading bot or the portfolio backtester.

Usage:
    # Live paper trading -- continuous loop (Ctrl+C to stop):
    python main.py
    python main.py --interval 5          # scan every 5 minutes
    python main.py --max-positions 5     # cap portfolio at 5 open positions
    python main.py --once                # single scan then exit

    # Portfolio backtest (historical simulation):
    python main.py --backtest --start 2025-03-26
    python main.py --backtest --start 2025-03-26 --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM

    # Backtest (skip rebuilding sentiment cache if already cached):
    python main.py --backtest --start 2025-03-26 --skip-cache-build

Requires environment variables:
    APCA_API_KEY_ID       -- Alpaca API key
    APCA_API_SECRET_KEY   -- Alpaca API secret
    APCA_API_BASE_URL     -- https://paper-api.alpaca.markets (paper trading)
"""

import sys
import os
import argparse
import warnings

# Suppress noisy third-party deprecation warnings that we cannot fix
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', message='.*utcnow.*', category=FutureWarning)

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_live(interval=15, max_positions=10, once=False):
    """Run the live paper-trading bot (src/bot.py __main__ logic)."""
    argv_backup = sys.argv[1:]
    sys.argv = [sys.argv[0],
                '--interval',      str(interval),
                '--max-positions', str(max_positions)]
    if once:
        sys.argv += ['--once']

    import runpy
    runpy.run_path(os.path.join(os.path.dirname(__file__), 'src', 'bot.py'),
                   run_name='__main__')

    sys.argv = [sys.argv[0]] + argv_backup


def run_backtest(start, tickers=None, skip_cache=False):
    """Run the portfolio backtester (src/backtest_portfolio.py __main__ logic)."""
    argv_backup = sys.argv[1:]
    sys.argv = [sys.argv[0], '--backtest-start', start]
    if tickers:
        sys.argv += ['--tickers', ','.join(tickers) if isinstance(tickers, list) else tickers]
    if skip_cache:
        sys.argv += ['--skip-cache-build']

    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), 'src', 'backtest_portfolio.py'),
        run_name='__main__')

    sys.argv = [sys.argv[0]] + argv_backup


def main():
    parser = argparse.ArgumentParser(
        description='LLM-Driven Trading Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                         # live trading, scan every 15 min
  python main.py --interval 5                           # live trading, scan every 5 min
  python main.py --max-positions 5                      # live trading, cap at 5 open positions
  python main.py --once                                 # live trading, single scan then exit
  python main.py --backtest --start 2025-03-26          # backtest (full S&P 500)
  python main.py --backtest --start 2025-03-26 \\
    --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM       # backtest, 7-ticker override
  python main.py --backtest --start 2025-03-26 \\
    --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM \\
    --skip-cache-build                                   # backtest, skip sentiment rebuild
        """
    )
    # ── backtest flags ────────────────────────────────────────────────────────
    parser.add_argument('--backtest', action='store_true',
                        help='Run portfolio backtest instead of live trading')
    parser.add_argument('--start', type=str, default='2025-03-26',
                        help='Backtest start date YYYY-MM-DD (default: 2025-03-26)')
    parser.add_argument('--tickers', type=str, nargs='+', default=None,
                        help='Ticker override for backtest (comma-separated, e.g. AAPL,MSFT,GOOGL)')
    parser.add_argument('--skip-cache-build', action='store_true',
                        help='Skip sentiment cache rebuild (use existing cached scores)')
    # ── live bot flags ────────────────────────────────────────────────────────
    parser.add_argument('--interval', type=int, default=15,
                        help='Minutes between live monitoring scans (default: 15)')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Max total open positions at any time (default: 10)')
    parser.add_argument('--once', action='store_true',
                        help='Live mode: run a single scan then exit (no continuous loop)')
    args = parser.parse_args()

    print('\n' + '=' * 65)
    print('  LLM-Driven Trading Bot  |  Student Capital Growth Strategy')
    print('  Model: DistilRoBERTa-financial  |  5-Criteria Funnel')
    print('=' * 65)

    if args.backtest:
        print(f'\n  Mode: BACKTEST  |  Start: {args.start}')
        tickers = args.tickers
        if tickers and len(tickers) == 1 and ',' in tickers[0]:
            tickers = tickers[0].split(',')
        run_backtest(args.start, tickers=tickers, skip_cache=args.skip_cache_build)
    else:
        print('\n  Mode: LIVE PAPER TRADING')
        run_live(interval=args.interval,
                 max_positions=args.max_positions,
                 once=args.once)


if __name__ == '__main__':
    main()
