"""
verify.py -- Walk-Forward Validation for grid_search_results.csv

Reads the top N unique parameter combinations from the grid search output CSV
and re-runs each on an out-of-sample (OOS) date window. Prints a side-by-side
table comparing in-sample vs OOS performance, with a PASS/FAIL verdict.

Rationale:
  The grid search is in-sample (trained on the backtest window). A result that
  holds on a different date range with a different market character provides
  stronger evidence of a genuine edge rather than curve-fitting.

  Default OOS window: 2024-01-01 -> 2025-03-25 (pre-tariff-crash bull market,
  a different regime from the post-crash recovery used in grid search).

Usage:
    # Build OOS sentiment caches (~5 min) then validate top 10 combos:
    python src/verify.py \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM \\
        --validation-start 2024-01-01 --validation-end 2025-03-25

    # If OOS caches already exist (faster):
    python src/verify.py \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM \\
        --validation-start 2024-01-01 --validation-end 2025-03-25 \\
        --skip-cache-build

    # Validate only top 5 combos:
    python src/verify.py \\
        --tickers AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRM --top-n 5 --skip-cache-build
"""

import os
import sys
import time
import argparse
import multiprocessing
import io
import contextlib
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Parameters that are set on the backtest_portfolio module for each combo run
# ---------------------------------------------------------------------------
PARAM_NAMES = [
    'ATR_MULTIPLIER',
    'TAKE_PROFIT_PCT',
    'SENTIMENT_FLOOR',
    'RSI_ENTRY_MIN',
    'MIN_HEADLINES',
    'REENTRY_COOLDOWN_DAYS',
    'TREND_FAIL_MIN_HOLD',
]

# Dedup key: RSI_ENTRY_MIN confirmed irrelevant in grid search (all values give
# identical results). Unique combos are identified by these three parameters only.
DEDUP_KEYS = ['ATR_MULTIPLIER', 'TAKE_PROFIT_PCT', 'SENTIMENT_FLOOR']


# ---------------------------------------------------------------------------
# Worker functions (module-level for pickle on Windows spawn)
# ---------------------------------------------------------------------------
_WORKER_DATA = {}


def _init_worker(data_bundle: dict):
    """Called once per worker process to receive pre-computed OOS data."""
    global _WORKER_DATA
    _WORKER_DATA = data_bundle


def _run_one(combo: dict) -> dict:
    """
    Run a single OOS simulation with the given parameter combination.
    combo contains both PARAM_NAMES keys and _is_* in-sample metric keys.
    Only PARAM_NAMES are patched onto the backtest_portfolio module.
    """
    import backtest_portfolio as bp

    for name in PARAM_NAMES:
        setattr(bp, name, combo[name])

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
        return {
            **combo,
            'oos_return':    float('nan'),
            'oos_sharpe':    float('nan'),
            'oos_win_rate':  float('nan'),
            'oos_n_trades':  0,
            'oos_drawdown':  float('nan'),
            'oos_stop_rate': float('nan'),
            'oos_error':     str(e),
            'oos_run_sec':   round(time.time() - t0, 1),
        }

    n_trades  = metrics['n_trades']
    n_winners = metrics['n_winners']
    n_stops   = metrics['n_stops']
    win_rate  = n_winners / n_trades if n_trades > 0 else 0.0
    stop_rate = n_stops   / n_trades if n_trades > 0 else 0.0

    return {
        **combo,
        'oos_return':    metrics['total_return'],
        'oos_sharpe':    metrics['sharpe'],
        'oos_win_rate':  win_rate,
        'oos_n_trades':  n_trades,
        'oos_drawdown':  metrics['max_drawdown'],
        'oos_stop_rate': stop_rate,
        'oos_error':     '',
        'oos_run_sec':   round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _verdict(oos_return: float, oos_sharpe: float) -> str:
    if oos_return != oos_return or oos_sharpe != oos_sharpe:  # NaN
        return 'ERROR'
    if oos_return > 0.05 and oos_sharpe > 0.5:
        return 'STRONG_PASS'
    if oos_return > 0.0 and oos_sharpe > 0.0:
        return 'PASS'
    if oos_return > 0.0:
        return 'MARGINAL'
    return 'FAIL'


def _load_unique_combos(csv_path: str, top_n: int) -> list:
    """
    Read grid_search_results.csv, deduplicate by DEDUP_KEYS, and return the
    top N unique combos by return UNION top N unique combos by Sharpe.
    Each returned dict has all PARAM_NAMES keys + _is_* in-sample metric keys.
    """
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    for col in PARAM_NAMES + ['total_return', 'sharpe']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from {csv_path}")

    is_metric_cols = ['total_return', 'sharpe', 'win_rate',
                      'n_trades', 'max_drawdown', 'stop_rate']

    seen_keys = set()
    unique_by_return = []
    for _, row in df.sort_values('total_return', ascending=False).iterrows():
        key = tuple(round(row[k], 4) for k in DEDUP_KEYS)
        if key not in seen_keys:
            seen_keys.add(key)
            combo = {p: row[p] for p in PARAM_NAMES}
            for col in is_metric_cols:
                combo[f'_is_{col}'] = row.get(col, float('nan'))
            unique_by_return.append(combo)
        if len(unique_by_return) >= top_n:
            break

    seen_keys_sharpe = set()
    unique_by_sharpe = []
    for _, row in df.sort_values('sharpe', ascending=False).iterrows():
        key = tuple(round(row[k], 4) for k in DEDUP_KEYS)
        if key not in seen_keys_sharpe:
            seen_keys_sharpe.add(key)
            combo = {p: row[p] for p in PARAM_NAMES}
            for col in is_metric_cols:
                combo[f'_is_{col}'] = row.get(col, float('nan'))
            unique_by_sharpe.append(combo)
        if len(unique_by_sharpe) >= top_n:
            break

    # Union: merge both lists, dedup by DEDUP_KEYS
    all_seen = set()
    merged = []
    for combo in unique_by_return + unique_by_sharpe:
        key = tuple(round(combo[k], 4) for k in DEDUP_KEYS)
        if key not in all_seen:
            all_seen.add(key)
            merged.append(combo)

    return merged


def _print_results_table(results: list, is_window: str, oos_window: str):
    hdr_is  = f'IN-SAMPLE ({is_window})'
    hdr_oos = f'OOS ({oos_window})'
    sep = '=' * 115
    print(f"\n{sep}")
    print(f"  WALK-FORWARD VALIDATION RESULTS")
    print(sep)
    print(f"  {'#':>3}  {'ATR':>5} {'TP%':>5} {'SENT':>5} | "
          f"{'--- ' + hdr_is + ' ---':^38} | "
          f"{'--- ' + hdr_oos + ' ---':^38} | VERDICT")
    print(f"  {'':>3}  {'':>5} {'':>5} {'':>5} | "
          f"{'RET%':>6} {'SHRP':>6} {'WIN%':>5} {'TRD':>4} {'DD%':>6} {'STOP%':>6} | "
          f"{'RET%':>6} {'SHRP':>6} {'WIN%':>5} {'TRD':>4} {'DD%':>6} {'STOP%':>6} |")
    print('  ' + '-' * 112)

    for i, row in enumerate(results, 1):
        is_ret   = row['_is_total_return']
        is_shrp  = row['_is_sharpe']
        is_win   = row['_is_win_rate']
        is_trd   = int(row['_is_n_trades']) if row['_is_n_trades'] == row['_is_n_trades'] else 0
        is_dd    = row['_is_max_drawdown']
        is_stop  = row['_is_stop_rate']

        oos_ret  = row['oos_return']
        oos_shrp = row['oos_sharpe']
        oos_win  = row['oos_win_rate']
        oos_trd  = int(row['oos_n_trades'])
        oos_dd   = row['oos_drawdown']
        oos_stop = row['oos_stop_rate']

        verdict = _verdict(oos_ret, oos_shrp)

        def fmt_ret(v):
            return f'{v*100:>+6.2f}' if v == v else '   NaN'
        def fmt_f(v):
            return f'{v:>6.2f}' if v == v else '   NaN'
        def fmt_pct(v):
            return f'{v*100:>5.1f}' if v == v else '  NaN'
        def fmt_dd(v):
            return f'{v*100:>6.1f}' if v == v else '   NaN'

        print(f"  {i:>3}  {row['ATR_MULTIPLIER']:>5.1f} {row['TAKE_PROFIT_PCT']*100:>4.0f}% "
              f"{row['SENTIMENT_FLOOR']:>5.1f} | "
              f"{fmt_ret(is_ret)} {fmt_f(is_shrp)} {fmt_pct(is_win)} {is_trd:>4} "
              f"{fmt_dd(is_dd)} {fmt_pct(is_stop)} | "
              f"{fmt_ret(oos_ret)} {fmt_f(oos_shrp)} {fmt_pct(oos_win)} {oos_trd:>4} "
              f"{fmt_dd(oos_dd)} {fmt_pct(oos_stop)} | {verdict}")

    print(sep)
    verdicts = [_verdict(r['oos_return'], r['oos_sharpe']) for r in results]
    passes   = sum(1 for v in verdicts if v in ('PASS', 'STRONG_PASS'))
    strong   = sum(1 for v in verdicts if v == 'STRONG_PASS')
    print(f"\n  Summary: {passes}/{len(results)} combos PASS or better  "
          f"({strong} STRONG_PASS)")
    if passes == 0:
        print("  ! All combos FAIL OOS — parameters are likely overfit to the in-sample window.")
    elif passes < len(results) // 2:
        print("  ! Majority fail OOS — treat in-sample results with caution.")
    else:
        print("  Majority hold OOS — parameters show evidence of genuine edge.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward validation using grid_search_results.csv'
    )
    parser.add_argument('--csv', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            'plots', 'grid_search_results.csv'),
                        help='Path to grid_search_results.csv')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of unique combos to validate (default: 10)')
    parser.add_argument('--in-sample-start', type=str, default='2025-03-26',
                        help='In-sample start date (for display only, default: 2025-03-26)')
    parser.add_argument('--in-sample-end', type=str, default='2026-03-30',
                        help='In-sample end date (for display only, default: 2026-03-30)')
    parser.add_argument('--validation-start', type=str, default='2024-01-01',
                        help='OOS validation start date (default: 2024-01-01)')
    parser.add_argument('--validation-end', type=str, default='2025-03-25',
                        help='OOS validation end date (default: 2025-03-25)')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Comma-separated ticker override')
    parser.add_argument('--inflation', type=float, default=0.035)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--skip-cache-build', action='store_true',
                        help='Skip building OOS sentiment caches (must already exist)')
    args = parser.parse_args()

    ticker_override = ([t.strip().upper() for t in args.tickers.split(',')]
                       if args.tickers else None)

    # ------------------------------------------------------------------
    # Load unique combos from CSV
    # ------------------------------------------------------------------
    if not os.path.exists(args.csv):
        print(f"  ERROR: CSV not found: {args.csv}")
        print(f"  Run src/optimize.py first to generate it.")
        sys.exit(1)

    combos = _load_unique_combos(args.csv, args.top_n)
    n_combos = len(combos)

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD VALIDATOR")
    print(f"{'='*65}")
    print(f"  CSV:              {args.csv}")
    print(f"  Unique combos:    {n_combos} "
          f"(top {args.top_n} by return + top {args.top_n} by Sharpe, deduped)")
    print(f"  In-sample:        {args.in_sample_start} -> {args.in_sample_end}")
    print(f"  OOS window:       {args.validation_start} -> {args.validation_end}")
    print(f"  Tickers:          {ticker_override or 'S&P 500 universe'}")
    print(f"  Workers:          {min(args.workers, n_combos)}")
    print(f"  Cache build:      {'SKIP (--skip-cache-build)' if args.skip_cache_build else 'ENABLED'}")

    print(f"\n  Top {n_combos} combos to validate:")
    print(f"  {'#':>3}  {'ATR':>5} {'TP%':>5} {'SENT':>5}  IS_RET%  IS_SHARPE")
    print('  ' + '-' * 45)
    for i, c in enumerate(combos, 1):
        print(f"  {i:>3}  {c['ATR_MULTIPLIER']:>5.1f} {c['TAKE_PROFIT_PCT']*100:>4.0f}% "
              f"{c['SENTIMENT_FLOOR']:>5.1f}  "
              f"{c['_is_total_return']*100:>+6.2f}%   {c['_is_sharpe']:>6.2f}")

    # ------------------------------------------------------------------
    # SETUP PHASE (once in main process, for OOS window)
    # ------------------------------------------------------------------
    import backtest_portfolio as bp
    from sentiment import SentimentAgent
    from data import NewsFetcher

    print(f"\n{'='*65}")
    print(f"  [Step 1/3] Historical pipeline as of {args.validation_start}")
    print(f"{'='*65}")

    ranked_candidates, _ = bp.run_historical_pipeline(
        args.validation_start,
        inflation_rate=args.inflation,
        ticker_override=ticker_override,
    )

    candidate_tickers = [c['ticker'] for c in ranked_candidates]
    price_data = bp.prefetch_prices(
        candidate_tickers, args.validation_start, args.validation_end)
    ranked_candidates = [c for c in ranked_candidates if c['ticker'] in price_data]

    sim_start = date.fromisoformat(args.validation_start)
    spy_dates = [
        d.date() if hasattr(d, 'date') else d
        for d in price_data['SPY'].index
        if (d.date() if hasattr(d, 'date') else d) >= sim_start
    ]

    tech_score_series, ma_crossover_series = bp.compute_tech_score_series(
        price_data, sim_start)
    rsi_series  = bp.compute_rsi_series(price_data, sim_start)
    atr_series  = bp.compute_atr_series(price_data, sim_start)

    print(f"\n{'='*65}")
    print(f"  [Step 2/3] OOS sentiment caches "
          f"({'loading existing' if args.skip_cache_build else 'building — this may take ~5 min'})")
    print(f"{'='*65}")

    os.environ.setdefault('APCA_API_KEY_ID',     os.getenv('APCA_API_KEY_ID', ''))
    os.environ.setdefault('APCA_API_SECRET_KEY', os.getenv('APCA_API_SECRET_KEY', ''))
    os.environ.setdefault('APCA_API_BASE_URL',
                          os.getenv('APCA_API_BASE_URL',
                                    'https://paper-api.alpaca.markets'))

    sentiment_agent = SentimentAgent()
    news_fetcher    = NewsFetcher()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cache_data = bp.build_sentiment_caches(
            ranked_candidates, spy_dates, sentiment_agent, news_fetcher,
            skip=args.skip_cache_build,
        )

    cache_lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    for line in cache_lines[-5:]:
        print(f"  {line}")

    print(f"\n  OOS setup complete.")

    data_bundle = {
        'ranked_candidates':   ranked_candidates,
        'price_data':          price_data,
        'cache_data':          cache_data,
        'backtest_start':      args.validation_start,
        'backtest_end':        args.validation_end,
        'tech_score_series':   tech_score_series,
        'ma_crossover_series': ma_crossover_series,
        'rsi_series':          rsi_series,
        'atr_series':          atr_series,
    }

    # ------------------------------------------------------------------
    # VALIDATION LOOP — parallel
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  [Step 3/3] Running {n_combos} OOS simulations...")
    print(f"{'='*65}\n")

    n_workers   = min(args.workers, n_combos)
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
            elapsed  = time.time() - t_start
            avg_sec  = elapsed / completed
            eta_sec  = avg_sec * (n_combos - completed)
            ret_str  = (f"{row['oos_return']*100:+.2f}%"
                        if row['oos_return'] == row['oos_return'] else 'ERR')
            shrp_str = (f"{row['oos_sharpe']:.2f}"
                        if row['oos_sharpe'] == row['oos_sharpe'] else 'ERR')
            print(f"  [{completed:>2}/{n_combos}]  ETA {eta_sec:.0f}s  "
                  f"OOS: return={ret_str} sharpe={shrp_str} "
                  f"trades={row.get('oos_n_trades', '?')}")

    total_elapsed = time.time() - t_start
    print(f"\n  Validation complete in {total_elapsed:.1f}s")

    # Sort by OOS return (desc) for display
    all_results.sort(key=lambda r: r['oos_return']
                     if r['oos_return'] == r['oos_return'] else -999,
                     reverse=True)

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------
    _print_results_table(
        all_results,
        is_window=f"{args.in_sample_start} -> {args.in_sample_end}",
        oos_window=f"{args.validation_start} -> {args.validation_end}",
    )

    # Save CSV
    output_dir  = bp._DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'verify_results.csv')

    rows_for_csv = []
    for row in all_results:
        csv_row = {p: row[p] for p in PARAM_NAMES}
        csv_row['is_return']   = row['_is_total_return']
        csv_row['is_sharpe']   = row['_is_sharpe']
        csv_row['is_win_rate'] = row['_is_win_rate']
        csv_row['is_n_trades'] = row['_is_n_trades']
        csv_row['is_drawdown'] = row['_is_max_drawdown']
        csv_row['is_stop_rate']= row['_is_stop_rate']
        csv_row['oos_return']  = row['oos_return']
        csv_row['oos_sharpe']  = row['oos_sharpe']
        csv_row['oos_win_rate']= row['oos_win_rate']
        csv_row['oos_n_trades']= row['oos_n_trades']
        csv_row['oos_drawdown']= row['oos_drawdown']
        csv_row['oos_stop_rate']=row['oos_stop_rate']
        csv_row['verdict']     = _verdict(row['oos_return'], row['oos_sharpe'])
        csv_row['oos_error']   = row.get('oos_error', '')
        rows_for_csv.append(csv_row)

    pd.DataFrame(rows_for_csv).to_csv(output_path, index=False, float_format='%.6f')
    print(f"\n  Results saved: {output_path}")
    print(f"\n  NOTE: OOS results are on {args.validation_start} -> {args.validation_end}.")
    print(f"  A PASS verdict confirms the parameter set is not solely curve-fit to the")
    print(f"  in-sample window. A FAIL does not invalidate the strategy — it means the")
    print(f"  specific parameter values may need further tuning or wider validation.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
