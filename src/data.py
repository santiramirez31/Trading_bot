import os
import warnings
import pandas as pd
import yfinance as yf
from alpaca_trade_api.rest import REST, TimeFrame

# Suppress yfinance calling deprecated pd.Timestamp.utcnow().
# Must come AFTER pandas import because pandas registers an "always" filter
# for Pandas4Warning at import time; inserting here puts our "ignore" at the
# front of the filter chain so it wins.
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*utcnow.*")


class MarketDataHandler:
    def __init__(self, api_key: str = None, api_secret: str = None, base_url: str = 'https://paper-api.alpaca.markets'):
        self.api_key = api_key or os.getenv('APCA_API_KEY_ID')
        self.api_secret = api_secret or os.getenv('APCA_API_SECRET_KEY')
        self.base_url = base_url
        self.api = None
        if self.api_key and self.api_secret:
            try:
                self.api = REST(self.api_key, self.api_secret, self.base_url)
            except Exception as e:
                print(f"Failed to initialize Alpaca API: {e}")

    def get_historical_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetches historical OHLCV data. Tries Alpaca first, falls back to yfinance.
        start and end formats: 'YYYY-MM-DD'
        """
        if self.api:
            try:
                bars = self.api.get_bars(ticker, TimeFrame.Day, start=start, end=end, feed='iex').df
                if not bars.empty:
                    bars = bars.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    return bars[['Open', 'High', 'Low', 'Close', 'Volume']]
            except Exception as e:
                print(f"Alpaca data fetch failed for {ticker}: {e}. Falling back to yfinance.")

        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            print(f"yfinance data fetch failed for {ticker}: {e}")

        return pd.DataFrame()

    def get_stocks_with_positive_return(self, tickers: list = None, min_return: float = 0.05, years: int = 5) -> list:
        """
        Stage 1 filter: screens tickers for long-term inflation-beating quality.
        Criteria:
          - Annualized return over `years` >= min_return (default 5% above inflation benchmark)
          - Positive Sharpe ratio proxy (total return > 0)
        Defaults to full S&P 500 if no tickers provided.
        Returns filtered list sorted by annualized return descending.
        """
        import datetime
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=years * 365)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        if tickers is None:
            print("Fetching S&P 500 company list from Wikipedia...")
            try:
                table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
                tickers = table[0]['Symbol'].tolist()
            except Exception as e:
                print(f"Failed to fetch S&P 500 list: {e}")
                return []

        print(f"Stage 1 -- Quality screen: {len(tickers)} tickers, {years}-year return >= {min_return*100:.0f}%...")

        try:
            df = yf.download(tickers, start=start_str, end=end_str, interval='1mo', progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                close_df = df['Close']
            else:
                close_df = df

            scored = []
            for ticker in tickers:
                if ticker not in close_df.columns:
                    continue
                ticker_data = close_df[ticker].dropna()
                if len(ticker_data) < 2:
                    continue
                first_price = float(ticker_data.iloc[0])
                last_price = float(ticker_data.iloc[-1])
                if first_price <= 0:
                    continue
                total_return = (last_price - first_price) / first_price
                annualized_return = (1 + total_return) ** (1 / years) - 1
                if annualized_return >= min_return and total_return > 0:
                    scored.append((ticker, annualized_return))

        except Exception as e:
            print(f"Stage 1 screening failed: {e}")
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        result = [t for t, _ in scored]
        print(f"Stage 1 complete: {len(result)} tickers passed quality screen.")
        return result

    def get_technical_candidates(self, tickers: list, lookback_days: int = 250, min_avg_volume: int = 1_000_000) -> list:
        """
        Stage 2 filter: applies technical trend and liquidity checks.
        Criteria:
          - Current price > 50-day MA  (short-term uptrend)
          - 50-day MA > 200-day MA     (long-term trend confirmation)
          - 20-day average volume > min_avg_volume (liquidity screen)
        Returns list of dicts sorted by trend strength (price/MA50 ratio desc):
          [{'ticker': str, 'price': float, 'ma50': float, 'ma200': float, 'avg_volume': float}, ...]
        """
        import datetime
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=lookback_days + 10)

        print(f"\nStage 2 -- Technical screen: {len(tickers)} tickers, MA50>MA200 + volume>{min_avg_volume/1e6:.0f}M...")

        candidates = []
        # Batch download for efficiency
        try:
            df = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                close_df = df['Close']
                volume_df = df['Volume']
            else:
                # Single ticker fallback
                close_df = df[['Close']]
                close_df.columns = tickers
                volume_df = df[['Volume']]
                volume_df.columns = tickers
        except Exception as e:
            print(f"Stage 2 batch download failed: {e}")
            return []

        for ticker in tickers:
            try:
                if ticker not in close_df.columns:
                    continue
                prices = close_df[ticker].dropna()
                volumes = volume_df[ticker].dropna()
                if len(prices) < 200:
                    continue

                price = float(prices.iloc[-1])
                ma50  = float(prices.rolling(50).mean().iloc[-1])
                ma200 = float(prices.rolling(200).mean().iloc[-1])
                avg_vol = float(volumes.tail(20).mean())

                if price > ma50 and ma50 > ma200 and avg_vol > min_avg_volume:
                    trend_strength = price / ma50  # how far above MA50
                    candidates.append({
                        'ticker': ticker,
                        'price': round(price, 2),
                        'ma50': round(ma50, 2),
                        'ma200': round(ma200, 2),
                        'avg_volume': int(avg_vol),
                        'trend_strength': round(trend_strength, 4),
                    })
            except Exception:
                continue

        candidates.sort(key=lambda x: x['trend_strength'], reverse=True)
        print(f"Stage 2 complete: {len(candidates)} tickers in confirmed uptrend with sufficient liquidity.")
        return candidates

    def get_sp500_universe(self) -> tuple:
        """
        Returns (tickers: list[str], sectors: dict[str, str]) from Wikipedia.
        sectors maps ticker -> GICS sector name.
        Uses a browser-like User-Agent to avoid 403 blocks.
        """
        import urllib.request, io
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8')
            tables = pd.read_html(io.StringIO(html))
            df = tables[0]
            tickers = df['Symbol'].tolist()
            # Fix tickers with dots (e.g. BRK.B -> BRK-B for yfinance)
            tickers = [t.replace('.', '-') for t in tickers]
            sector_col = 'GICS Sector' if 'GICS Sector' in df.columns else df.columns[3]
            raw_sectors = dict(zip(df['Symbol'], df[sector_col]))
            sectors = {t.replace('.', '-'): s for t, s in raw_sectors.items()}
            print(f"S&P 500 universe: {len(tickers)} tickers across {len(set(sectors.values()))} sectors.")
            return tickers, sectors
        except Exception as e:
            print(f"Failed to fetch S&P 500 universe: {e}")
            return [], {}

    def quality_screen_10yr(self, tickers: list, inflation_rate: float = 0.035,
                             min_excess_pct: float = 0.05, risk_free: float = 0.04,
                             as_of_date=None) -> list:
        """
        Criterion 1: 10-year annualized return > (inflation + min_excess) AND Sharpe ratio > 0.

        as_of_date: optional str or datetime — caps the download end date for historical backtests.
                    When None (default), uses today (live bot behaviour, fully backward-compatible).

        Returns list of dicts sorted by ann_return desc:
        [{'ticker': str, 'ann_return': float, 'sharpe': float}, ...]
        """
        import datetime
        min_return = inflation_rate + min_excess_pct  # default: 8.5%
        end_date   = pd.Timestamp(as_of_date) if as_of_date else datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=10 * 365 + 30)  # 10yr + buffer

        print(f"\nCriterion 1 -- Quality screen: {len(tickers)} tickers | 10yr return > {min_return*100:.1f}% AND Sharpe > 0...")

        try:
            df = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'), interval='1mo',
                             progress=False, auto_adjust=True)
            close_df = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
        except Exception as e:
            print(f"  Quality screen download failed: {e}")
            return []

        passed = []
        for ticker in tickers:
            try:
                if ticker not in close_df.columns:
                    continue
                series = close_df[ticker].dropna()
                if len(series) < 60:  # need at least 5yr of monthly data
                    continue
                first, last = float(series.iloc[0]), float(series.iloc[-1])
                if first <= 0:
                    continue
                years = len(series) / 12.0
                total_return = (last - first) / first
                ann_return = (1 + total_return) ** (1 / years) - 1
                # Sharpe on monthly returns
                monthly_returns = series.pct_change().dropna()
                if len(monthly_returns) < 24:
                    continue
                ann_vol = float(monthly_returns.std()) * (12 ** 0.5)
                sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0.0
                if ann_return >= min_return and sharpe > 0:
                    passed.append({'ticker': ticker, 'ann_return': round(ann_return, 4), 'sharpe': round(sharpe, 3)})
            except Exception:
                continue

        passed.sort(key=lambda x: x['ann_return'], reverse=True)
        print(f"  Criterion 1 complete: {len(passed)} / {len(tickers)} passed.")
        return passed

    def screen_liquidity_risk_trend(self, quality_results: list,
                                     min_volume: int = 1_000_000,
                                     min_price: float = 5.0,
                                     max_volatility: float = 0.50,
                                     max_drawdown: float = -0.60,
                                     as_of_date=None) -> list:
        """
        Applies Criteria 4, 5 as binary filters and computes C2 (tech) score for ranking.
        Uses ONE batch download of 1yr daily data.

        as_of_date: optional str or datetime — caps the download end date for historical backtests.
                    When None (default), uses today (live bot behaviour, fully backward-compatible).

        C4 (Liquidity): avg 20d volume > min_volume AND price > min_price
        C5 (Risk):      90d annualized downside deviation < max_volatility AND 1y max drawdown > max_drawdown
        C2 (Trend score): tech_score = (price/ma50 - 1) + (ma50/ma200 - 1) -- used for ranking only

        Returns list of dicts sorted by rank_score (high = strong trend + low risk):
        {ticker, sector(if provided), price, ma50, ma200, avg_volume,
         ann_return, sharpe, volatility, max_drawdown, tech_score, risk_score, rank_score}
        """
        import datetime
        tickers = [r['ticker'] for r in quality_results]
        quality_map = {r['ticker']: r for r in quality_results}

        end_date   = pd.Timestamp(as_of_date) if as_of_date else datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=360)  # 360 calendar days ~= 252 trading days (enough for MA200)

        print(f"\nCriteria 4+5 -- Liquidity & risk screen: {len(tickers)} tickers...")

        try:
            df = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'), interval='1d',
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                close_df  = df['Close']
                volume_df = df['Volume']
            else:
                close_df  = df[['Close']]; close_df.columns  = tickers
                volume_df = df[['Volume']]; volume_df.columns = tickers
        except Exception as e:
            print(f"  Liquidity/risk screen download failed: {e}")
            return []

        c4_fail_vol, c4_fail_price, c5_fail_vol, c5_fail_dd = 0, 0, 0, 0
        survivors = []

        for ticker in tickers:
            try:
                if ticker not in close_df.columns:
                    continue
                prices  = close_df[ticker].dropna()
                volumes = volume_df[ticker].dropna()
                if len(prices) < 200:
                    continue

                price     = float(prices.iloc[-1])
                avg_vol   = float(volumes.tail(20).mean())
                ma50      = float(prices.rolling(50).mean().iloc[-1])
                ma200     = float(prices.rolling(200).mean().iloc[-1])

                # --- Criterion 4: Liquidity ---
                if avg_vol < min_volume:
                    c4_fail_vol += 1; continue
                if price < min_price:
                    c4_fail_price += 1; continue

                # --- Criterion 5: Downside deviation (90d annualized) ---
                # Uses only negative daily returns (semi-deviation) so stocks with
                # large upswings are not penalised for their gains — only downside
                # risk matters here. Threshold is set lower than the old std-dev
                # threshold (~0.25 ≈ equivalent to ~0.35 std-dev for a symmetric dist).
                daily_returns_90 = prices.pct_change().dropna().tail(90)
                negative_returns = daily_returns_90[daily_returns_90 < 0]
                volatility = float(negative_returns.std()) * (252 ** 0.5) if len(negative_returns) > 1 else 0.0
                if volatility >= max_volatility:
                    c5_fail_vol += 1; continue

                # --- Criterion 5: Max drawdown (1yr) ---
                prices_1yr  = prices.tail(252)
                roll_max    = prices_1yr.cummax()
                drawdown_series = (prices_1yr - roll_max) / roll_max
                max_dd_val  = float(drawdown_series.min())
                if max_dd_val <= max_drawdown:
                    c5_fail_dd += 1; continue

                # --- C2 tech score (for ranking, not binary filter) ---
                price_gap  = (price - ma50)  / ma50   # positive = price above MA50
                ma_gap     = (ma50  - ma200) / ma200  # positive = MA50 above MA200
                tech_score = price_gap + ma_gap       # higher = stronger trend (can be negative)

                # --- C5 risk score (0 to 1, higher = safer) ---
                vol_score  = max(0.0, 1.0 - volatility / max_volatility)
                dd_score   = max(0.0, 1.0 + max_dd_val / abs(max_drawdown))
                risk_score = (vol_score + dd_score) / 2.0

                # --- Combined rank score ---
                tech_norm  = min(max(tech_score, 0.0) / 0.15, 1.0)  # normalize, cap at 1.0
                rank_score = 0.5 * tech_norm + 0.5 * risk_score

                q = quality_map[ticker]
                survivors.append({
                    'ticker':       ticker,
                    'price':        round(price, 2),
                    'ma50':         round(ma50, 2),
                    'ma200':        round(ma200, 2),
                    'avg_volume':   int(avg_vol),
                    'ann_return':   q['ann_return'],
                    'sharpe':       q['sharpe'],
                    'volatility':   round(volatility, 4),
                    'max_drawdown': round(max_dd_val, 4),
                    'tech_score':   round(tech_score, 4),
                    'risk_score':   round(risk_score, 4),
                    'rank_score':   round(rank_score, 4),
                })
            except Exception:
                continue

        survivors.sort(key=lambda x: x['rank_score'], reverse=True)
        print(f"  C4 rejected: {c4_fail_vol} (low volume) + {c4_fail_price} (penny stock)")
        print(f"  C5 rejected: {c5_fail_vol} (high volatility) + {c5_fail_dd} (deep drawdown)")
        print(f"  Criteria 4+5 complete: {len(survivors)} survivors, ranked by risk+trend score.")
        return survivors


class NewsFetcher:
    def __init__(self, api_key: str = None, api_secret: str = None, base_url: str = 'https://paper-api.alpaca.markets'):
        self.api_key = api_key or os.getenv('APCA_API_KEY_ID')
        self.api_secret = api_secret or os.getenv('APCA_API_SECRET_KEY')
        self.base_url = base_url
        self.api = None
        if self.api_key and self.api_secret:
            try:
                self.api = REST(self.api_key, self.api_secret, self.base_url)
            except Exception as e:
                print(f"Failed to initialize Alpaca API for News: {e}")

    def get_recent_news(self, ticker: str, limit: int = 15) -> list:
        """
        Fetches recent news headlines for a ticker using the Alpaca News API.
        Returns a list of dicts with 'headline', 'summary', 'url', 'created_at', 'source'.
        Most recent items come first (index 0 = newest).
        'source' is the publisher domain (e.g. 'benzinga.com') used for source whitelist filtering.
        """
        if self.api:
            try:
                news = self.api.get_news(ticker, limit=limit)
                return [
                    {
                        "headline":   n.headline,
                        "summary":    n.summary,
                        "url":        n.url,
                        "created_at": n.created_at,
                        "source":     getattr(n, 'source', '') or '',
                    }
                    for n in news
                ]
            except Exception as e:
                print(f"Alpaca news fetch failed for {ticker}: {e}.")
        else:
            print("Alpaca API not initialized. Cannot fetch news.")

        return []
