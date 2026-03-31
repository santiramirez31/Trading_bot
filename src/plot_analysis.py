"""
plot_analysis.py  --  Per-ticker trade & sentiment visualisation

Reads:
  plots/portfolio_trades.csv   -- trade log from backtest_portfolio.py
  data/sentiment_cache_*.csv   -- daily LLM sentiment scores
  (yfinance for price data)

Produces:
  plots/trade_analysis.png     -- one row per traded ticker:
      * Close price line
      * 50-day MA (cyan dashed)  -> gate: price must be above MA50 for entry
      * Hold-period shading (green = profitable, red = loss)
      * Buy (green triangle up) / sell (red triangle down / gold star) markers
      * Sentiment score overlay on right axis (filled area + smoothed line)
        -> green area = bullish (>5), red area = bearish (<5)
        -> up arrows = improving momentum, down arrows = worsening momentum
      * Portfolio equity vs SPY on the first row

Usage:
    python src/plot_analysis.py
    python src/plot_analysis.py --backtest-start 2025-03-26
"""

import os, argparse
from datetime import date

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
_SRC_DIR    = os.path.dirname(__file__)
_ROOT_DIR   = os.path.dirname(_SRC_DIR)
_TRADES_CSV = os.path.join(_ROOT_DIR, 'plots', 'portfolio_trades.csv')
_CACHE_DIR  = os.path.join(_ROOT_DIR, 'data')
_OUT_DIR    = os.path.join(_ROOT_DIR, 'plots')

INITIAL_CAPITAL = 100_000.0
NEUTRAL_SCORE   = 5.0
MA50_COL        = '#00e5ff'   # cyan  -- 50-day MA
PRICE_COL       = '#c8cdd6'
SENT_LINE_COL   = '#80c8ff'
# --------------------------------------------------------------------------- #


# ── data loaders ────────────────────────────────────────────────────────────

def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['entry_date'] = pd.to_datetime(df['entry_date']).dt.date
    df['exit_date']  = pd.to_datetime(df['exit_date']).dt.date
    return df


def load_sentiment(ticker: str) -> pd.DataFrame:
    path = os.path.join(_CACHE_DIR, f'sentiment_cache_{ticker.upper()}.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    df = df.set_index(date_col)
    if 'momentum' not in df.columns:
        df['momentum'] = 0
    return df


def fetch_prices(tickers: list, sim_start: str, end: str,
                 ma_warmup_days: int = 300) -> dict:
    """
    Downloads OHLCV including `ma_warmup_days` extra calendar days before
    `sim_start` so that the 50-day MA is fully warmed up on the first
    simulation day.  Returns dict: ticker -> DataFrame (full history).
    """
    all_tickers = list(dict.fromkeys(tickers + ['SPY']))
    dl_start = (pd.Timestamp(sim_start)
                - pd.Timedelta(days=ma_warmup_days + 60)).strftime('%Y-%m-%d')
    print(f"  Fetching prices from {dl_start} to {end} (MA warm-up included)...")
    raw = yf.download(all_tickers, start=dl_start, end=end,
                      progress=False, auto_adjust=True)
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in all_tickers:
            try:
                df = raw.xs(t, axis=1, level=1)[['Open','High','Low','Close']].dropna()
                if len(df) > 20:
                    out[t] = df
            except Exception:
                pass
    else:
        t = all_tickers[0]
        df = raw[['Open','High','Low','Close']].dropna()
        if len(df) > 20:
            out[t] = df
    return out


def compute_mas(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame with MA50 (50-day moving average) aligned to prices_df index."""
    closes = prices_df['Close']
    return pd.DataFrame({
        'MA50': closes.rolling(50, min_periods=50).mean(),
    }, index=prices_df.index)


# ── equity reconstruction ────────────────────────────────────────────────────

def reconstruct_equity(trades: pd.DataFrame, price_data: dict,
                       sim_start: str, initial_capital: float) -> pd.Series:
    if 'SPY' not in price_data:
        return pd.Series(dtype=float)

    sim_ts  = pd.Timestamp(sim_start)
    spy_idx = price_data['SPY'].index
    dates   = sorted(d for d in spy_idx if d >= sim_ts)

    cash     = initial_capital
    positions: list = []
    equity   = {}

    all_pos = trades.to_dict('records')

    for d in dates:
        d_date = d.date() if hasattr(d, 'date') else d
        for pos in all_pos:
            if pos['entry_date'] == d_date and pos not in positions:
                cash -= int(pos['shares']) * float(pos['entry_price'])
                positions.append(pos)
        for pos in positions[:]:
            if pos['exit_date'] == d_date:
                cash += int(pos['shares']) * float(pos['exit_price'])
                positions.remove(pos)
        pv = 0.0
        for pos in positions:
            t = pos['ticker']
            if t in price_data and d in price_data[t].index:
                pv += int(pos['shares']) * float(price_data[t].loc[d, 'Close'])
            else:
                pv += int(pos['shares']) * float(pos['entry_price'])
        equity[d_date] = cash + pv

    return pd.Series(equity)


# ── axis styling helper ──────────────────────────────────────────────────────

def _style_ax(ax, grid_col, txt_col, interval_months=2):
    ax.set_facecolor('#181c24')
    ax.tick_params(colors=txt_col, labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.yaxis.grid(True, color=grid_col, lw=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(grid_col)


# ── main plot ────────────────────────────────────────────────────────────────

def plot(trades_csv: str, backtest_start: str, backtest_end: str, out_path: str):

    trades = load_trades(trades_csv)
    if trades.empty:
        print("No trades found.")
        return

    tickers_traded = [t for t in trades['ticker'].unique() if t != 'SPY']
    tickers_traded = sorted(tickers_traded,
                             key=lambda t: -len(trades[trades['ticker'] == t]))
    print(f"  Tickers: {tickers_traded}")

    price_data = fetch_prices(tickers_traded, backtest_start, backtest_end)
    sim_ts     = pd.Timestamp(backtest_start)

    # ── figure layout ────────────────────────────────────────────────────────
    n_tickers = len(tickers_traded)
    n_rows    = 1 + n_tickers
    fig_h     = 5.0 + n_tickers * 5.2
    dark_bg   = '#0e1117'
    grid_col  = '#2a2d35'
    txt_col   = '#e0e0e0'

    fig = plt.figure(figsize=(22, fig_h), facecolor=dark_bg)
    gs  = fig.add_gridspec(n_rows, 1, hspace=0.65,
                           left=0.055, right=0.925, top=0.97, bottom=0.045)

    # ── ROW 0 : portfolio equity vs SPY ──────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0])
    equity = reconstruct_equity(trades, price_data, backtest_start, INITIAL_CAPITAL)

    if not equity.empty:
        eq_ts  = [pd.Timestamp(d) for d in equity.index]
        eq_pct = equity.values / INITIAL_CAPITAL * 100 - 100
        ax_eq.plot(eq_ts, eq_pct, color='#f0a500', lw=2.2,
                   label='Strategy', zorder=3)
        ax_eq.fill_between(eq_ts, eq_pct, 0,
                            where=eq_pct >= 0, color='#f0a500', alpha=0.12)
        ax_eq.fill_between(eq_ts, eq_pct, 0,
                            where=eq_pct <  0, color='#ff4444', alpha=0.12)

    if 'SPY' in price_data:
        spy = price_data['SPY']['Close']
        spy = spy[spy.index >= sim_ts]
        spy_pct = (spy / spy.iloc[0] - 1) * 100
        ax_eq.plot(spy.index, spy_pct.values, color='#5b9bd5', lw=1.6,
                   label='SPY buy & hold', zorder=2)

    ax_eq.axhline(0, color=grid_col, lw=0.9, ls='--')
    final_ret = eq_pct[-1] if not equity.empty else 0
    spy_ret   = float(spy_pct.iloc[-1]) if 'SPY' in price_data else 0
    ax_eq.set_title(
        f'Portfolio Return vs SPY  ({backtest_start} to {backtest_end})   '
        f'Strategy: {final_ret:+.1f}%   SPY: {spy_ret:+.1f}%',
        color=txt_col, fontsize=11, pad=7, loc='left')
    ax_eq.set_ylabel('Return %', color=txt_col, fontsize=9)
    _style_ax(ax_eq, grid_col, txt_col)
    ax_eq.legend(fontsize=9, framealpha=0.3, facecolor='#2a2d35',
                  edgecolor=grid_col, labelcolor=txt_col)

    # ── ROWS 1+ : per-ticker panels ───────────────────────────────────────────
    for row_i, ticker in enumerate(tickers_traded):
        ax_p = fig.add_subplot(gs[row_i + 1])
        ax_s = ax_p.twinx()
        ax_s.set_facecolor('#181c24')

        t_trades     = trades[trades['ticker'] == ticker].copy()
        sentiment_df = load_sentiment(ticker)
        prices_df    = price_data.get(ticker)

        # ── moving averages ───────────────────────────────────────────────────
        if prices_df is not None and not prices_df.empty:
            mas = compute_mas(prices_df)
            # Clip to simulation window for display
            sim_prices = prices_df[prices_df.index >= sim_ts]
            sim_mas    = mas[mas.index >= sim_ts]

            ax_p.plot(sim_prices.index, sim_prices['Close'],
                      color=PRICE_COL, lw=1.3, zorder=2, alpha=0.9,
                      label='Close')
            ax_p.plot(sim_mas.index, sim_mas['MA50'],
                      color=MA50_COL, lw=1.1, ls='--', zorder=3, alpha=0.85,
                      label='MA 50')

            # Small legend inside the panel
            ax_p.legend(fontsize=6.5, framealpha=0.25,
                         facecolor='#2a2d35', edgecolor=grid_col,
                         labelcolor=txt_col, loc='upper left',
                         ncol=2, markerscale=0.8)

        # ── hold-period shading ───────────────────────────────────────────────
        # Colour by outcome (profit/loss), not by exit mechanism.
        # A trailing stop that fires after the price rose is still a winning trade.
        for _, tr in t_trades.iterrows():
            trade_pnl = tr['exit_price'] / tr['entry_price'] - 1
            shade = '#00c85518' if trade_pnl >= 0 else '#ff444418'
            ax_p.axvspan(pd.Timestamp(tr['entry_date']),
                         pd.Timestamp(tr['exit_date']),
                         color=shade, zorder=1)

        # ── buy / sell markers ────────────────────────────────────────────────
        for _, tr in t_trades.iterrows():
            entry_ts = pd.Timestamp(tr['entry_date'])
            exit_ts  = pd.Timestamp(tr['exit_date'])
            pnl      = float(tr.get('pnl_pct',
                             (tr['exit_price']/tr['entry_price']-1)*100))

            ax_p.scatter(entry_ts, tr['entry_price'],
                         marker='^', s=120, color='#00e676',
                         zorder=6, linewidths=0)

            if tr['exit_reason'] == 'TAKE_PROFIT':
                sell_col, mkr, sz = '#ffd700', '*', 220
                ax_p.annotate(f"+{pnl:.1f}%",
                              xy=(exit_ts, tr['exit_price']),
                              xytext=(5, 7), textcoords='offset points',
                              fontsize=8, color='#ffd700',
                              fontweight='bold', zorder=7)
            elif tr['exit_reason'] == 'STOP_LOSS':
                sell_col = '#ff7070' if pnl >= 0 else '#ff3333'
                mkr, sz  = 'v', 100
                if pnl > 0.3:      # trailing stop locked in a gain -- label it
                    ax_p.annotate(f"+{pnl:.1f}%",
                                  xy=(exit_ts, tr['exit_price']),
                                  xytext=(5, -12), textcoords='offset points',
                                  fontsize=7.5, color='#ff7070', zorder=7)
            else:
                sell_col, mkr, sz = '#aaaaaa', 'o', 60

            ax_p.scatter(exit_ts, tr['exit_price'],
                         marker=mkr, s=sz, color=sell_col,
                         zorder=6, linewidths=0)

        # ── sentiment overlay (right axis) ────────────────────────────────────
        if not sentiment_df.empty:
            s_news = sentiment_df[
                sentiment_df.get('headline_count',
                pd.Series(0, index=sentiment_df.index)) > 0
            ]
            if not s_news.empty:
                s_dates  = [pd.Timestamp(d) for d in s_news.index]
                s_scores = s_news['score'].astype(float)
                s_smooth = s_scores.rolling(5, min_periods=1, center=True).mean()

                ax_s.fill_between(s_dates, NEUTRAL_SCORE, s_smooth.values,
                                  where=s_smooth.values >= NEUTRAL_SCORE,
                                  color='#00c853', alpha=0.22, zorder=1,
                                  interpolate=True)
                ax_s.fill_between(s_dates, s_smooth.values, NEUTRAL_SCORE,
                                  where=s_smooth.values < NEUTRAL_SCORE,
                                  color='#ff5252', alpha=0.22, zorder=1,
                                  interpolate=True)
                ax_s.plot(s_dates, s_smooth.values,
                          color=SENT_LINE_COL, lw=1.3, zorder=3, alpha=0.9)

                # Momentum arrows: ↑ improving, ↓ worsening
                if 'momentum' in s_news.columns:
                    for d, mom in s_news['momentum'].items():
                        score = float(s_news.loc[d, 'score'])
                        ts    = pd.Timestamp(d)
                        if mom == 1:
                            ax_s.annotate(
                                '', xy=(ts, score + 0.7),
                                xytext=(ts, score + 0.05),
                                arrowprops=dict(arrowstyle='->', lw=1.1,
                                                color='#00e676'),
                                zorder=5)
                        elif mom == -1:
                            ax_s.annotate(
                                '', xy=(ts, score - 0.7),
                                xytext=(ts, score - 0.05),
                                arrowprops=dict(arrowstyle='->', lw=1.1,
                                                color='#ff5252'),
                                zorder=5)

        ax_s.axhline(NEUTRAL_SCORE, color='#ffffff20', lw=0.9, ls=':', zorder=2)
        ax_s.set_ylim(0.5, 10.5)
        ax_s.set_yticks([2, 4, 5, 6, 8])
        ax_s.set_yticklabels(['2', '4', '5 neutral', '6', '8'], fontsize=6.5)
        ax_s.set_ylabel('Sentiment (0–10)', color=SENT_LINE_COL,
                         fontsize=8, labelpad=4)
        ax_s.tick_params(axis='y', colors=SENT_LINE_COL, labelsize=6.5)
        for spine in ax_s.spines.values():
            spine.set_color(grid_col)

        # ── panel title with trade stats ──────────────────────────────────────
        n_tp  = (t_trades['exit_reason'] == 'TAKE_PROFIT').sum()
        n_sl  = (t_trades['exit_reason'] == 'STOP_LOSS').sum()
        n_tot = len(t_trades)
        wins  = (t_trades['exit_price'] / t_trades['entry_price'] > 1).sum()

        # Total return: sum of (pnl_pct * shares * entry_price) across all trades
        # = net dollars gained/lost, divided by total capital deployed
        t_trades = t_trades.copy()
        t_trades['trade_pnl_$'] = ((t_trades['exit_price'] - t_trades['entry_price'])
                                    * t_trades['shares'])
        t_trades['deployed_$']  = t_trades['entry_price'] * t_trades['shares']
        total_pnl   = t_trades['trade_pnl_$'].sum()
        total_dep   = t_trades['deployed_$'].sum()
        ticker_ret  = (total_pnl / total_dep * 100) if total_dep > 0 else 0.0
        ret_sign = '+' if ticker_ret >= 0 else ''
        ret_col  = '#00e676' if ticker_ret >= 0 else '#ff5252'

        ax_p.set_title(
            f'{ticker}   {n_tot} trades  |  '
            f'Take-profits: {n_tp}  Stop-losses: {n_sl}  |  '
            f'Win rate: {wins}/{n_tot} ({wins/n_tot*100:.0f}%)',
            color=txt_col, fontsize=10, pad=6, loc='left')

        # Return figure in a coloured annotation top-right of the price panel
        ax_p.annotate(
            f'Return: {ret_sign}{ticker_ret:.1f}%',
            xy=(1.0, 1.01), xycoords='axes fraction',
            ha='right', va='bottom',
            fontsize=10, fontweight='bold', color=ret_col)

        ax_p.set_ylabel('Price ($)', color=txt_col, fontsize=8)
        _style_ax(ax_p, grid_col, txt_col)

    # ── global legend ─────────────────────────────────────────────────────────
    legend_elements = [
        # Price / MA lines
        Line2D([0],[0], color=PRICE_COL,  lw=1.5, label='Close price'),
        Line2D([0],[0], color=MA50_COL,   lw=1.5, ls='--',
               label='50-day MA  (entry gate: price must be above)'),
        mpatches.Patch(facecolor='#00e67620',
                       label='Profit'),
        mpatches.Patch(facecolor='#ff444420',
                       label='Loss'),
        # Trade markers
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#00e676',
               markersize=9, linewidth=0, label='Buy entry'),
        Line2D([0],[0], marker='v', color='w', markerfacecolor='#ff3333',
               markersize=9, linewidth=0, label='Stop-loss exit'),
        Line2D([0],[0], marker='*', color='w', markerfacecolor='#ffd700',
               markersize=12, linewidth=0, label='Take-profit exit (+10%)'),
        # Hold shading
        mpatches.Patch(facecolor='#00c85530',
                       label='Hold period — ended in take-profit'),
        mpatches.Patch(facecolor='#ff444430',
                       label='Hold period — ended in stop-loss'),
        # Sentiment
        Line2D([0],[0], color=SENT_LINE_COL, lw=1.5,
               label='Sentiment score (5-day smoothed, right axis)'),
        mpatches.Patch(facecolor='#00c85535',
                       label='Sentiment > 5 (bullish news)'),
        mpatches.Patch(facecolor='#ff525235',
                       label='Sentiment < 5 (bearish news)'),
        # Momentum arrows
        Line2D([0],[0], marker=r'$\uparrow$', color='#00e676', markersize=10,
               linewidth=0, label='Sentiment momentum: IMPROVING day-over-day'),
        Line2D([0],[0], marker=r'$\downarrow$', color='#ff5252', markersize=10,
               linewidth=0, label='Sentiment momentum: WORSENING day-over-day'),
    ]

    fig.legend(handles=legend_elements,
               loc='lower center', ncol=3, fontsize=8.5,
               framealpha=0.35, facecolor='#1e2230',
               edgecolor=grid_col, labelcolor=txt_col,
               bbox_to_anchor=(0.5, 0.002),
               borderpad=0.8, handlelength=1.6)

    fig.patch.set_facecolor(dark_bg)
    os.makedirs(_OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor=dark_bg)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trades-csv',      default=_TRADES_CSV)
    parser.add_argument('--backtest-start',  default='2025-03-26')
    parser.add_argument('--backtest-end',    default=date.today().strftime('%Y-%m-%d'))
    parser.add_argument('--out', default=os.path.join(_OUT_DIR, 'trade_analysis.png'))
    args = parser.parse_args()

    print(f"\n  Loading trades: {args.trades_csv}")
    plot(args.trades_csv, args.backtest_start, args.backtest_end, args.out)
