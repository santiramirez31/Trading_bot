"""
optimize.py -- Grid Search Parameter Optimizer for backtest_portfolio.py

Runs hundreds of backtest simulations across parameter combinations, reusing
all expensive setup (price download, sentiment cache load, ATR/RSI computation)
in the main process and distributing simulation work across CPU cores.

Architecture:
  1. Setup once in main process (model load + yfinance + series computation)
  2. multiprocessing.Pool sends pre-computed data to N workers via initializer
  3. Each worker monkey-patches its own backtest_portfolio module globals and
     runs bp.run_simulation() independently
  4. Results collected into plots/grid_search_results.csv

NOTE ON OVERFITTING:
  This grid search trains on the full backtest period (in-sample). The top-ranked
  parameter set may be over-fit to this specific 12-month window. To validate,
  take the top 3 sets from grid_search_results.csv and re-run backtest_portfolio.py
  on a different date window (e.g. --backtest-start set 6 months later).

Usage:
    python src/optimize.py --backtest-start 2025-03-26 \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM --grid quick --dry-run
    python src/optimize.py --backtest-start 2025-03-26 \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM --grid quick
    python src/optimize.py --backtest-start 2025-03-26 \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM --grid medium
    python src/optimize.py --backtest-start 2025-03-26 \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM --grid full
    python src/optimize.py --backtest-start 2025-03-26 \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM --grid comprehensive
"""

import os
import sys
import time
import argparse
import itertools
import multiprocessing
import io
import contextlib
from datetime import date, timedelta

import pandas as pd

# Add src/ to path so imports work when run from project root
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------
GRIDS = {
    # 3×3×3×2 = 54 runs  (~3 min parallel)
    'quick': {
        'ATR_MULTIPLIER':        [2.0, 2.5, 3.0],
        'TAKE_PROFIT_PCT':       [0.10, 0.15, 0.20],
        'SENTIMENT_FLOOR':       [6.5, 7.0, 7.5],
        'RSI_ENTRY_MIN':         [30, 35],
        'MIN_HEADLINES':         [5],
        'REENTRY_COOLDOWN_DAYS': [5],
        'TREND_FAIL_MIN_HOLD':   [5],
    },
    # 5×5×4×4 = 400 runs  (~9 min parallel)
    'medium': {
        'ATR_MULTIPLIER':        [1.5, 2.0, 2.5, 3.0, 3.5],
        'TAKE_PROFIT_PCT':       [0.08, 0.10, 0.12, 0.15, 0.20],
        'SENTIMENT_FLOOR':       [6.0, 6.5, 7.0, 7.5],
        'RSI_ENTRY_MIN':         [25, 30, 35, 40],
        'MIN_HEADLINES':         [5],
        'REENTRY_COOLDOWN_DAYS': [5],
        'TREND_FAIL_MIN_HOLD':   [5],
    },
    # 4×4×3×3×2×3 = 864 runs  (~18 min parallel)
    'full': {
        'ATR_MULTIPLIER':        [2.0, 2.5, 3.0, 3.5],
        'TAKE_PROFIT_PCT':       [0.10, 0.12, 0.15, 0.20],
        'SENTIMENT_FLOOR':       [6.5, 7.0, 7.5],
        'RSI_ENTRY_MIN':         [25, 30, 35],
        'MIN_HEADLINES':         [3, 5],
        'REENTRY_COOLDOWN_DAYS': [3, 5, 7],
        'TREND_FAIL_MIN_HOLD':   [5],
    },
    # 4×4×3×3×2×3×4 = 3456 runs  (~65 min parallel)
    'comprehensive': {
        'ATR_MULTIPLIER':        [2.0, 2.5, 3.0, 3.5],
        'TAKE_PROFIT_PCT':       [0.10, 0.12, 0.15, 0.20],
        'SENTIMENT_FLOOR':       [6.5, 7.0, 7.5],
        'RSI_ENTRY_MIN':         [25, 30, 35],
        'MIN_HEADLINES':         [3, 5],
        'REENTRY_COOLDOWN_DAYS': [3, 5, 7],
        'TREND_FAIL_MIN_HOLD':   [3, 5, 7, 10],
    },
}

# Baseline (v5b) for reference in output
BASELINE = {
    'ATR_MULTIPLIER': 2.5,
    'TAKE_PROFIT_PCT': 0.10,
    'SENTIMENT_FLOOR': 7.0,
    'RSI_ENTRY_MIN': 35,
    'MIN_HEADLINES': 5,
    'REENTRY_COOLDOWN_DAYS': 5,
    'TREND_FAIL_MIN_HOLD': 5,
    'total_return': 0.0144,
    'sharpe': -0.57,
    'max_drawdown': -0.043,
    'win_rate': 0.44,
    'n_trades': 34,
    'stop_rate': 0.53,
}

# ---------------------------------------------------------------------------
# Worker functions (must be module-level for pickle on Windows spawn)
# ---------------------------------------------------------------------------
_WORKER_DATA = {}


def _init_worker(data_bundle: dict):
    """Called once per worker process to load shared pre-computed data."""
    global _WORKER_DATA
    _WORKER_DATA = data_bundle


def _run_one(combo: dict) -> dict:
    """Run a single simulation with the given parameter combination."""
    import backtest_portfolio as bp

    # Monkey-patch this worker's module globals
    for name, val in combo.items():
        setattr(bp, name, val)

    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            results = bp.run_simulation(
                _WORKER_DATA['ranked_candidates'],
                _WORKER_DATA['price_data'],
                _WORKER_DATA['cache_data'],
                _WORKER_DATA['backtest_start'],
                _WORKER_DATA['backtest_end'],
                initial_capital=bp.INITIAL_CAPITAL,
                tech_score_series=_WORKER_DATA['tech_score_series'],
                ma_crossover_series=_WORKER_DATA['ma_crossover_series'],
                rsi_series=_WORKER_DATA['rsi_series'],
                atr_series=_WORKER_DATA['atr_series'],
            )
            metrics = bp.compute_metrics(results)
    except Exception as e:
        # Return a sentinel row so the grid run doesn't crash
        return {**combo,
                'total_return': float('nan'), 'ann_return': float('nan'),
                'sharpe': float('nan'), 'max_drawdown': float('nan'),
                'ann_vol': float('nan'), 'alpha': float('nan'),
                'win_rate': float('nan'), 'n_trades': 0,
                'n_stops': 0, 'n_targets': 0, 'n_open': 0,
                'stop_rate': float('nan'), 'run_sec': round(time.time() - t0, 1),
                'error': str(e)}

    n_trades = metrics['n_trades']
    n_winners = metrics['n_winners']
    n_stops = metrics['n_stops']
    win_rate = n_winners / n_trades if n_trades > 0 else 0.0
    stop_rate = n_stops / n_trades if n_trades > 0 else 0.0

    return {
        **combo,
        'total_return':  metrics['total_return'],
        'ann_return':    metrics['ann_return'],
        'sharpe':        metrics['sharpe'],
        'max_drawdown':  metrics['max_drawdown'],
        'ann_vol':       metrics['ann_vol'],
        'alpha':         metrics['alpha'],
        'win_rate':      win_rate,
        'n_trades':      n_trades,
        'n_stops':       n_stops,
        'n_targets':     metrics['n_targets'],
        'n_open':        metrics['n_open'],
        'stop_rate':     stop_rate,
        'run_sec':       round(time.time() - t0, 1),
        'error':         '',
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _print_top20(df: pd.DataFrame, sort_col: str, label: str, param_names: list):
    top = df.dropna(subset=[sort_col]).nlargest(20, sort_col)
    print(f"\n{'='*100}")
    print(f"  TOP 20 — sorted by {label}")
    print(f"{'='*100}")
    # Dynamic header based on which params were varied
    p_hdr = ' '.join(f'{p[:6]:>7}' for p in param_names)
    print(f"  {'#':>3}  {p_hdr}   {'RET%':>6} {'SHRP':>6} {'DD%':>6} "
          f"{'WIN%':>5} {'TRD':>4} {'STOP%':>6} {'NOTE'}")
    print('  ' + '-' * 97)
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        p_vals = ' '.join(f'{row[p]:>7.2f}' for p in param_names)
        note = ''
        if row.get('n_trades', 0) < 20:
            note = '! <20 trades'
        # Flag extreme parameter values
        grid_def = GRIDS.get(_CURRENT_GRID, {})
        for p in param_names:
            vals = grid_def.get(p, [row[p]])
            if len(vals) > 1 and row[p] in (min(vals), max(vals)):
                note = note or '! extreme value'
                break
        print(f"  {rank:>3}  {p_vals}   "
              f"{row['total_return']*100:>+6.2f} {row['sharpe']:>6.2f} "
              f"{row['max_drawdown']*100:>6.1f} {row['win_rate']*100:>5.1f} "
              f"{int(row['n_trades']):>4} {row['stop_rate']*100:>6.1f}  {note}")


_CURRENT_GRID = 'quick'  # updated in main before printing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _CURRENT_GRID

    parser = argparse.ArgumentParser(
        description='Grid search optimizer for backtest_portfolio.py'
    )
    parser.add_argument('--backtest-start', type=str,
                        default=(date.today() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Backtest start date YYYY-MM-DD (default: 1 year ago)')
    parser.add_argument('--backtest-end', type=str,
                        default=date.today().strftime('%Y-%m-%d'),
                        help='Backtest end date YYYY-MM-DD (default: today)')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated ticker override (e.g. AAPL,MSFT,GOOGL,...)')
    parser.add_argument('--inflation', type=float, default=0.035)
    parser.add_argument('--grid', type=str, default='quick',
                        choices=['quick', 'medium', 'full', 'comprehensive'],
                        help='quick=54 | medium=400 | full=864 | comprehensive=3456')
    parser.add_argument('--workers', type=int, default=10,
                        help='Parallel worker processes (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: plots/grid_search_results.csv)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print grid size and ETA, then exit without running')
    args = parser.parse_args()

    _CURRENT_GRID = args.grid
    backtest_start = args.backtest_start
    backtest_end   = args.backtest_end
    ticker_override = ([t.strip().upper() for t in args.tickers.split(',')]
                       if args.tickers else None)

    # Build grid
    grid_def = GRIDS[args.grid]
    param_names  = list(grid_def.keys())
    param_values = list(grid_def.values())
    combos = [dict(zip(param_names, vals))
              for vals in itertools.product(*param_values)]
    total_runs = len(combos)

    # Estimate timing
    est_sec_per_run = 12
    n_workers = min(args.workers, total_runs)
    est_parallel_sec = (total_runs / n_workers) * est_sec_per_run + 120
    est_seq_sec      = total_runs * est_sec_per_run + 120

    print(f"\n{'='*65}")
    print(f"  GRID SEARCH OPTIMIZER — {args.grid.upper()}")
    print(f"{'='*65}")
    print(f"  Grid:         {total_runs} combinations")
    print(f"  Parameters:   {', '.join(param_names)}")
    for p in param_names:
        vals = grid_def[p]
        if len(vals) > 1:
            print(f"    {p}: {vals}")
    print(f"  Workers:      {n_workers} (of {args.workers} requested)")
    print(f"  Est. time:    {est_parallel_sec/60:.0f} min parallel  "
          f"({est_seq_sec/60:.0f} min sequential)")
    print(f"  Backtest:     {backtest_start} -> {backtest_end}")
    print(f"  Tickers:      {ticker_override or 'S&P 500 universe'}")

    if args.dry_run:
        print(f"\n  DRY RUN — exiting without running simulations.")
        return

    # ------------------------------------------------------------------
    # SETUP PHASE (once in main process)
    # ------------------------------------------------------------------
    import backtest_portfolio as bp
    from sentiment import SentimentAgent
    from data import NewsFetcher

    ranked_candidates, _ = bp.run_historical_pipeline(
        backtest_start,
        inflation_rate=args.inflation,
        ticker_override=ticker_override,
    )

    candidate_tickers = [c['ticker'] for c in ranked_candidates]
    price_data = bp.prefetch_prices(candidate_tickers, backtest_start, backtest_end)
    ranked_candidates = [c for c in ranked_candidates if c['ticker'] in price_data]

    sim_start = date.fromisoformat(backtest_start)
    spy_dates = [
        d.date() if hasattr(d, 'date') else d
        for d in price_data['SPY'].index
        if (d.date() if hasattr(d, 'date') else d) >= sim_start
    ]

    tech_score_series, ma_crossover_series = bp.compute_tech_score_series(
        price_data, sim_start)
    rsi_series = bp.compute_rsi_series(price_data, sim_start)
    atr_series = bp.compute_atr_series(price_data, sim_start)

    # Sentiment: always skip cache build in optimizer (caches must already exist)
    os.environ.setdefault('APCA_API_KEY_ID',     os.getenv('APCA_API_KEY_ID', ''))
    os.environ.setdefault('APCA_API_SECRET_KEY', os.getenv('APCA_API_SECRET_KEY', ''))
    os.environ.setdefault('APCA_API_BASE_URL',
                          os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'))

    sentiment_agent = SentimentAgent()
    news_fetcher    = NewsFetcher()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cache_data = bp.build_sentiment_caches(
            ranked_candidates, spy_dates, sentiment_agent, news_fetcher,
            skip=True,
        )

    print(f"\n  Setup complete. Starting grid search with {n_workers} workers...\n")

    data_bundle = {
        'ranked_candidates':  ranked_candidates,
        'price_data':         price_data,
        'cache_data':         cache_data,
        'backtest_start':     backtest_start,
        'backtest_end':       backtest_end,
        'tech_score_series':  tech_score_series,
        'ma_crossover_series': ma_crossover_series,
        'rsi_series':         rsi_series,
        'atr_series':         atr_series,
    }

    # ------------------------------------------------------------------
    # GRID LOOP — parallel
    # ------------------------------------------------------------------
    all_results = []
    completed   = 0
    t_start     = time.time()

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(data_bundle,),
    ) as pool:
        for row in pool.imap_unordered(_run_one, combos):
            all_results.append(row)
            completed += 1
            elapsed    = time.time() - t_start
            avg_sec    = elapsed / completed
            eta_sec    = avg_sec * (total_runs - completed)
            if completed % 5 == 0 or completed == total_runs:
                ret_str = (f"{row['total_return']*100:+.2f}%"
                           if row['total_return'] == row['total_return'] else 'ERR')
                shrp_str = (f"{row['sharpe']:.2f}"
                            if row['sharpe'] == row['sharpe'] else 'ERR')
                print(f"  [{completed:>4}/{total_runs}]  ETA {eta_sec/60:.1f}m  "
                      f"last: return={ret_str} sharpe={shrp_str} "
                      f"trades={row.get('n_trades', '?')}")

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values('total_return', ascending=False).reset_index(drop=True)

    output_dir  = bp._DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = (args.output if args.output
                   else os.path.join(output_dir, 'grid_search_results.csv'))
    df_sorted.to_csv(output_path, index=False, float_format='%.6f')

    total_elapsed = time.time() - t_start
    print(f"\n  Grid search complete in {total_elapsed/60:.1f} min")
    print(f"  Results saved: {output_path}  ({len(df_sorted)} rows)")

    # Varied params only (for concise table headers)
    varied_params = [p for p in param_names if len(grid_def[p]) > 1]
    if not varied_params:
        varied_params = param_names

    _print_top20(df_sorted, 'total_return', 'Total Return', varied_params)
    _print_top20(df_sorted, 'sharpe',       'Sharpe Ratio', varied_params)

    # Baseline reference
    print(f"\n  BASELINE (v5b): return={BASELINE['total_return']*100:+.2f}%  "
          f"sharpe={BASELINE['sharpe']:.2f}  "
          f"max_dd={BASELINE['max_drawdown']*100:.1f}%  "
          f"win={BASELINE['win_rate']*100:.0f}%  "
          f"trades={BASELINE['n_trades']}  "
          f"stop_rate={BASELINE['stop_rate']*100:.0f}%")

    # Overfitting warnings
    print(f"\n  OVERFITTING WARNINGS:")
    print(f"  - Results are in-sample only (trained on {backtest_start} -> {backtest_end})")
    print(f"  - With ~34 trades/run, win-rate CI is ±17pp — top result may be noise")
    print(f"  - To validate: re-run backtest_portfolio.py with top params on a "
          f"different --backtest-start")
    thin = df_sorted[df_sorted['n_trades'] < 20]
    if not thin.empty:
        print(f"  - {len(thin)} combinations had <20 trades (marked ! in tables above)")


if __name__ == '__main__':
    multiprocessing.freeze_support()   # Required for Windows PyInstaller builds
    main()
