import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date, timezone
from pandas.tseries.offsets import BDay
import numpy as np
from scipy.stats import linregress
import json
from app.utils.config import AppConfig, STALE_THRESHOLD, INTERVAL_MAX_LOOKBACK
from app.utils.time_debug import timed
from app.utils.storage_utils import PortfolioDataManager
import logging
logger = logging.getLogger(__name__)

# To compare the P/B ratio of a stock to the five biggest companies by market cap, I use the benchmark below
# TODO automate the choice of companies, and add the remaining sectors
SECTOR_BENCHMARK_MAP = {
    'Technology': ['AAPL', 'MSFT','NVDA','TSM','AVGO'],
    'Financial Services': ['JPM', 'BRK-A','MA','BAC','V'],
    'Industrials': ['GE', 'CAT', 'RTX', 'SIE.DE', 'AIR.PA'],
    'Utilities': ['NEE', 'IBDSF', 'DOGEF', 'ENLAY', 'CEG'],
    'Healthcare': ['LLY', 'JNJ', 'AZN', 'UNH', 'ROG.SW'],
    'Real Estate': ['WELL', 'PLD', 'AMT', 'EQIX', 'SPG'],
    'Communication Services': ['GOOGL', 'META', 'TCEHY', 'NFLX', 'SFTBY'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'LVMUY', 'BABAF', 'HD'],
}

class FinanceDataManager:
    """
    Owns all yfinance I/O for one asset category (e.g. 'stocks', 'crypto').
    """
    
    DIV_CAGR_YEARS = 10
    BENCHMARK_REFRESH_DAYS = 7

    STALE_THRESHOLD = {
        "1m": timedelta(minutes=1), 
        "2m": timedelta(minutes=2),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(minutes=60),
        "90m": timedelta(minutes=90),
        "1d": timedelta(days=1),
    }

    INTERVAL_MAX_LOOKBACK = {
        "1m": timedelta(days=59), # safe margin under yfinance's ~60d cap for sub-daily
        "2m": timedelta(days=59),
        "5m": timedelta(days=59),
        "15m": timedelta(days=59),
        "30m": timedelta(days=59),
        "1h":  timedelta(days=59), 
        "90m": timedelta(days=59),
        "1d":  timedelta(days=365),
    }

    def __init__(self, cache_dir: str, category_name: str, config: AppConfig, portfolio_store: PortfolioDataManager):
        self.config = config
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True) 

        self.category = category_name
        self._portfolio_store = portfolio_store

        # Pull constants from config module
        self.STALE_THRESHOLD = STALE_THRESHOLD
        self.INTERVAL_MAX_LOOKBACK = INTERVAL_MAX_LOOKBACK

        # Derived paths — computed once, used everywhere
        self._metrics_path = os.path.join(cache_dir, f"{category_name}_metrics_static.json")
        self._fx_path      = os.path.join(cache_dir, "last_fetch_date.json")

        # In-memory caches
        self._hist_prices: dict[str, pd.DataFrame] = {}   # keyed by interval
        self._static_metrics: dict = {}
        self._usd_eur: float | None = None
        self._chf_eur: float | None = None

        self._weekend_sync_done = False

    def _get_price_path(self, interval: str) -> str:
        return os.path.join(self.cache_dir, f"{self.category}_price_history_{interval}.csv")

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #
    @timed
    def get_metrics(self, tickers: list[str], interval: str, force=False) -> list[dict]:
        """Main entry point. Returns metrics list for the requested tickers."""
        logger.debug(f"get_metrics called for interval {interval}, type(self)={type(self)}")
        self._ensure_fx_rates()
        self._ensure_prices(tickers, interval, force)
        return self._ensure_metrics(tickers, interval, force)

    def remove_ticker(self, ticker: str, interval: str):
        """Remove a ticker from both the CSV and the metrics JSON."""
        self._drop_from_csv(ticker, interval)
        self._drop_from_json(ticker)

    # ------------------------------------------------------------------ #
    #  Private helpers — each does ONE thing                              #
    # ------------------------------------------------------------------ #

    ### Orchestrators 
    def _ensure_fx_rates(self): # TODO Automatic currency fetching. Currently focuses on USD and CHF
        """Fetch FX rates at most once per day, cache in memory too."""
        if self._usd_eur is not None:
            return
        
        #date_cache_path = os.path.join(self.cache_dir, f"last_fetch_date.json")
        
        # Default fallback values
        # exchange_data = {
        #     "last_call_fx": "1900-01-01",
        #     "usd_eur_rate": 1.0,
        #     "chf_eur_rate": 1.0
        # }

        exchange_data = self._portfolio_store.get_fx_cache()
        last_call_fx = datetime.strptime(exchange_data["last_call_fx"], "%Y-%m-%d").date()

        # # Save new values
        # if os.path.exists(date_cache_path) and os.path.getsize(date_cache_path) > 0:
        #     try:
        #         with open(date_cache_path, 'r') as f:
        #             exchange_data.update(json.load(f))
        #         logger.info(f"\nLoaded last exchange rate: {exchange_data["last_call_fx"]}")
        #     except (json.JSONDecodeError, IOError) as e:
        #         logger.warning(f"Date cache file corrupted or empty, ignoring: {e}")
        # else:
        #     logger.warning("\nDate cache file is 0 bytes, ignoring.")
            
        # # Get last time function was called
        # last_call_fx = datetime.strptime(exchange_data["last_call_fx"], "%Y-%m-%d").date()
        logger.debug("last_call_fx: {last_call_fx}")
        
        # Lazy loading logic (once per day)
        if date.today()>last_call_fx:
            logger.info("Fetching exhange rates.")        

            try:
                # Fetch EUR/USD rate
                eur_usd_hist = yf.Ticker("EURUSD=X").history(period="1d")
                if not eur_usd_hist.empty and eur_usd_hist["Close"].dropna().iloc[-1] > 0:
                    exchange_data["usd_eur_rate"] = 1 / eur_usd_hist["Close"].dropna().iloc[-1] # Reciprocal (want EUR)
                
                # Fetch EUR/CHF rate
                eur_chf_hist = yf.Ticker("EURCHF=X").history(period="1d")
                if not eur_chf_hist.empty and eur_chf_hist["Close"].dropna().iloc[-1] > 0:
                    exchange_data["chf_eur_rate"] = 1 / eur_chf_hist["Close"].dropna().iloc[-1] # Reciprocal (want EUR)
                    
                exchange_data["last_call_fx"] = date.today().isoformat()
                #self._save_json(date_cache_path, exchange_data)
                self._portfolio_store.save_fx_cache(exchange_data)
                
            except Exception as e:
                logger.error(f"Error fetching exchange rates: {e}. Using last known/default rates.")
                
        else:
            logger.info("No need to update exchange rates for today.")
            
        self._usd_eur = exchange_data["usd_eur_rate"]
        self._chf_eur = exchange_data["chf_eur_rate"]

        logger.info(f"Rates fetched: USD/EUR = {self._usd_eur:.4f}, CHF/EUR = {self._chf_eur:.4f}")

    @timed
    def _ensure_prices( self, tickers: list[str], interval: str,
                        force: bool = False, target_start: datetime = None) -> None:
        logger.debug(f"[_ensure_prices] interval={interval}, tickers={tickers}")
        if interval not in self._hist_prices:
            df = self._load_csv(self._get_price_path(interval))
            self._hist_prices[interval] = self._normalize_tz(df)
            logger.debug(f"[_ensure_prices] loaded CSV, columns={list(df.columns)}")

        df = self._hist_prices[interval]
        logger.debug(f"[_ensure_prices] cached df columns={list(df.columns)}")

        new_tickers = [t for t in tickers if t not in df.columns]
        logger.debug(f"[_ensure_prices] new_tickers={new_tickers}")
        if new_tickers:
            df = self._add_tickers(df, new_tickers, interval)
            logger.debug(f"[_ensure_prices] after _add_tickers, columns={list(df.columns)}")

        if target_start:
            df = self._backfill_if_needed(df, target_start,
                                        tickers, interval)

        if self._is_stale(df, interval) or force: 
            df = self._refresh(df, tickers, interval)

        if df is not self._hist_prices[interval]: # only write if changed
            self._save_csv(df, interval)
            self._hist_prices[interval] = df
        
        return df

    def _ensure_metrics(self, tickers: list[str], interval: str, force: bool) -> list[dict]:
        if not self._static_metrics:
            self._static_metrics = self._load_json(self._metrics_path, default={})

        # Slow path — only when needed
        needs_fetch = force or not all(t in self._static_metrics for t in tickers)
        if needs_fetch:
            ticker_objs = yf.Tickers(" ".join(tickers))
            changed = False
            for ticker in tickers:
                if ticker not in self._static_metrics or force:
                    self._static_metrics[ticker] = self._compute_single_ticker(
                        ticker, ticker_objs.tickers[ticker], interval
                    )
                    sector = self._static_metrics[ticker].get("Sector")
                    if sector:
                        self._static_metrics[ticker]["Sector_PB_Benchmark"] = \
                            self._get_sector_benchmark(sector)
                    changed = True
            if changed:
                self._save_json(self._metrics_path, self._static_metrics)

        # Always inject fresh quote from price cache
        df = self._hist_prices.get(interval, pd.DataFrame())
        quotes_changed = False
        for ticker in tickers:
            if ticker not in self._static_metrics:
                continue
            if not df.empty and ticker in df.columns:
                last_price = float(df[ticker].dropna().iloc[-1])
                currency = self._static_metrics[ticker].get("Currency", "EUR")
                rate = self._usd_eur if currency == 'USD' else \
                    self._chf_eur if currency == 'CHF' else 1.0
                if rate:
                    self._static_metrics[ticker]["Quote"]     = last_price
                    self._static_metrics[ticker]["Quote_EUR"] = round(last_price * rate, 4)
                    quotes_changed = True

        if quotes_changed:
            self._save_json(self._metrics_path, self._static_metrics)

        return [self._static_metrics[t] for t in tickers if t in self._static_metrics]

    ### Workers
    def _add_tickers(self, df: pd.DataFrame, new_tickers: list[str], interval: str) -> pd.DataFrame:
        logger.debug(f"[_add_tickers] downloading {new_tickers} at {interval}")
        start = df.index.min() if not df.empty else datetime.now() - self.INTERVAL_MAX_LOOKBACK.get(interval, timedelta(days=59))
        new_prices = self._download_prices(new_tickers, interval, start, datetime.now())
        return self._merge(new_prices, df)

    def _backfill_if_needed(self, df: pd.DataFrame, target_start: datetime, 
                            tickers: list[str], interval: str) -> pd.DataFrame:
        first = df.index.min().tz_localize(None) if not df.empty else None
        target_ts = pd.to_datetime(target_start).tz_localize(None)
        if first is not None and target_ts >= first:
            return df  # Nothing to backfill
        logger.info(f"Backfilling history from {target_ts} to {first or datetime.now()}")
        end = first if first is not None else pd.Timestamp.now().tz_localize(None)
        gap_prices = self._download_prices(tickers, interval, target_ts, end)
        return self._merge(gap_prices, df) if not gap_prices.empty else df

    @timed
    def _refresh(self, df: pd.DataFrame, tickers: list[str], interval: str) -> pd.DataFrame:
        start = df.index.max() if not df.empty else datetime.now() - self.INTERVAL_MAX_LOOKBACK.get(interval, timedelta(days=59))
        new_prices = self._download_prices(tickers, interval, start, datetime.now())
        return self._merge(new_prices, df) if not new_prices.empty else df

    def _normalize_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert tz-aware index to naive UTC. Never strips without converting."""
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df
    
    @timed
    def _download_prices(self, tickers: list[str], interval: str, 
                     start: datetime, end: datetime) -> pd.DataFrame:
        """Download close prices for each ticker individually and join into one DataFrame."""
        yf_logger = logging.getLogger("yfinance")
        previous_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        # Download each ticker individually to avoid cross-timezone misalignment (avoid batch download)
        frames = []
        try:
            for ticker in tickers:
                ticker_start = start
                data = yf.download(ticker, start=ticker_start, end=end,
                                interval=interval, auto_adjust=True, progress=False)
                if not data.empty:
                    prices = self._extract_close(data, [ticker])
                    prices = self._normalize_tz(prices)
                    frames.append(prices)
        finally:
            yf_logger.setLevel(previous_level)

        if not frames:
            return pd.DataFrame()

        result = frames[0]
        for frame in frames[1:]:
            result = result.join(frame, how='outer')
        return result

    def _extract_close(self, raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # Always use xs regardless of ticker count since yfinance now always returns MultiIndex
                prices = raw.xs('Close', axis=1, level=0)
                # xs with level=0 returns a DataFrame with tickers as columns
                # but for a single ticker it may return a Series so we normalise it
                if isinstance(prices, pd.Series):
                    prices = prices.to_frame(name=tickers[0])
            else:
                # Fallback for older yfinance or already-processed data
                prices = raw[['Close']].rename(columns={'Close': tickers[0]})
            prices.columns = [str(c) for c in prices.columns]  # flatten any tuple labels
            return prices
        except KeyError as e:
            logger.error(f"Could not extract Close prices: {e}. Columns were: {raw.columns.tolist()}")
            return pd.DataFrame()

    def _merge(self, new: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
        """Merge two price DataFrames, removing duplicates and sorting by index."""
        if existing.empty:
            return new
        if new.empty:
            return existing
        merged = pd.concat([existing, new]).sort_index()
        merged = merged.ffill().groupby(level=0).last()   # dedup, keep latest value
        merged = merged.reindex(sorted(merged.columns), axis=1)  # sort columns alphabetically
        merged.index.name = 'Datetime'
        return merged

    def _compute_single_ticker(self, ticker: str, ticker_handle, interval: str) -> dict:
        """Compute all metrics for one ticker. Assumes FX rates and prices are already loaded."""
        # Initialisation
        data = {"Ticker": ticker, "Currency": "N/A", "Quote":0.0, "Quote_EUR": 0.0, "P/E": 0.0, "Fwd_P/E": 0.0, 
            "P/B": 0.0, "PEG": 0.0, "Earnings_Growth": 0.0, "Div_Yield": 0.0, "Div_CAGR": 0.0, 
            "Latest_Div_EUR": 0.0, "Months_Paid": [0]*12, "Sector": "N/A", "PayoutRatio": 0.0}

        logger.info(f"\nProcessing data for {ticker}")
        try:
            info = ticker_handle.info
            currency = info.get('currency', 'EUR')
            rate = self._usd_eur if currency == 'USD' else self._chf_eur if currency == 'CHF' else 1.0

            df = self._hist_prices.get(interval, pd.DataFrame())
            if ticker in df.columns:
                data["Quote"] = df[ticker].dropna().iloc[-1]
                data["Quote_EUR"] = float(data["Quote"] * rate)

            self._fill_valuation(data, info, currency, ticker)      # P/E, P/B, PEG etc.
            self._fill_earnings_growth(data, ticker, ticker_handle)
            self._fill_dividends(data, ticker_handle, currency)

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

        return data
    
    def _fill_valuation(self, data: dict, info, currency: str, ticker: str) -> dict:
        # Update dictionary with the latest fetch
        data.update({
            "Currency": currency,
            "Sector": info.get('sector'),
            "PayoutRatio": round(info['payoutRatio'], 4) if isinstance(info.get('payoutRatio'), (int, float)) else 0.0,
            "P/E": round(info['trailingPE'], 2) if isinstance(info.get('trailingPE'), (int, float)) else 0.0,
            "Fwd_P/E": round(info['forwardPE'], 2) if isinstance(info.get('forwardPE'), (int, float)) else 0.0,
            "P/B": round(info['priceToBook'], 2) if isinstance(info.get('priceToBook'), (int, float)) else 0.0
        })
                
        
        if data["P/E"] == 0.0:
            eps_ttm = info.get('trailingEps') # If trailingPE is 0.0 (missing), try to calculate it manually using trailingEps
            logger.debug("eps: ", eps_ttm)
            # Use currentPrice from info or fallback to cached quote
            price_native = info.get('currentPrice') or (data["Quote"] if data["Quote"] != 0 else None)
            logger.debug("price_native: ", price_native)
            if eps_ttm and price_native:
                data["P/E"] = round(price_native / eps_ttm, 2)
                logger.info(f"Calculated fallback P/E for {ticker}: {data['P/E']}")

        return data
    
    def _fill_earnings_growth(self, data: dict, ticker: str, ticker_handle: yf.Ticker) -> dict:
        # Get income statement for earnings growth 
        try:
            income = ticker_handle.income_stmt
            if not income.empty:
                
                net_income = income.loc["Net Income"].T.to_frame(name="Earnings").sort_index()
                # TODO Include operating income later
                #op_income = income.loc["Operating Income"].T.to_frame(name="Operating Income").sort_index()
                last_net_income = net_income["Earnings"].iloc[-1]
                penultimate_net_income = net_income["Earnings"].iloc[-2]
            
                # Annual earnings growth (YoY)
                if not np.isnan(last_net_income):
                    net_growth = last_net_income / abs(penultimate_net_income) - 1
                else:
                    # Fallback necessary because Yahoo Finance may take time to update financial data.
                    logger.info(f"Last net income for {ticker} absent. Using previous two incomes.")
                    penpenultimate_net_income = net_income["Earnings"].iloc[-3]
                    net_growth = penultimate_net_income / abs(penpenultimate_net_income) - 1
                    
                data["Earnings_Growth"] = round(net_growth, 4) if isinstance(net_growth, (int, float)) else "N/A"
                
                if isinstance(data["P/E"], float):
                    data["PEG"] = round(data["P/E"] / (net_growth * 100), 2)
        except Exception:
            logger.warning("Couldn't get Income Statement!")

        return data

    def _fill_dividends(self, data: dict, ticker_handle: yf.Ticker, currency: str) -> dict:
        
        one_year_ago = datetime.today() - timedelta(days=365)
        n_years_ago = datetime.today() - timedelta(days=self.DIV_CAGR_YEARS * 365)
        # Get dividend data
        actions = ticker_handle.actions
        
        if not actions.empty and 'Dividends' in actions.columns and actions['Dividends'].sum() > 0:            
            divs = actions[actions["Dividends"] > 0]["Dividends"].tz_localize(None) # Forget timezone
            if not divs.empty:
                # TTM Yield
                ttm_divs = divs[divs.index >= one_year_ago]
                
                # I focus on euro from here
                if not ttm_divs.empty and data["Quote_EUR"] != 0.0:
                    ttm_sum_eur = ttm_divs.sum() * (self._usd_eur if currency == 'USD' else self._chf_eur if currency == 'CHF' else 1)
                    data["Div_Yield"] = round(ttm_sum_eur / data["Quote_EUR"], 4)

                # Months paid
                for m_date in ttm_divs.index:
                    data["Months_Paid"][m_date.month - 1] = 1
                    
                # Latest dividend amount
                latest_div = divs.iloc[-1]
                if isinstance(latest_div, (int, float)):
                    if currency == 'USD':
                        latest_div = latest_div * self._usd_eur
                    elif currency == 'CHF':
                        latest_div = latest_div * self._chf_eur
                    data["Latest_Div_EUR"] = round(latest_div, 4)
                    
                # Growth rate (CAGR via log-linear regression)
                divs_filtered = divs[divs.index >= n_years_ago]
                data["Div_CAGR"] = calculate_growth_rate(divs_filtered)
        return data

    def _get_sector_benchmark(self, sector: str) -> float:
        """Returns avg P/B for sector proxy tickers. Refreshes every BENCHMARK_REFRESH_DAYS."""
        # Dedicated file — no longer sharing with FX cache
        bench_path = os.path.join(self.cache_dir, "sector_benchmarks.json")
        bench_data = self._load_json(bench_path, default={"dates": {}, "values": {}})

        last_str = bench_data["dates"].get(sector, "1900-01-01")
        last_date = datetime.strptime(last_str, "%Y-%m-%d").date()
        is_stale = (date.today() - last_date) > timedelta(days=self.BENCHMARK_REFRESH_DAYS)
        is_missing = sector not in bench_data["values"]

        if is_stale:
            logger.info(f"\n{self.BENCHMARK_REFRESH_DAYS} days passed. Updating P/B benchmark for {sector}...")
        elif is_missing:
            logger.info(f"\nSector {sector} is missing. Updating its P/B benchmark")

        if not (is_stale or is_missing):
            return bench_data["values"].get(sector, 0.0)

        proxies = SECTOR_BENCHMARK_MAP.get(sector)
        if not proxies:
            return 0.0

        pb_values = []
        for p in yf.Tickers(" ".join(proxies)).tickers.values():
            try:
                pb = p.info.get("priceToBook")
                if isinstance(pb, (int, float)):
                    pb_values.append(pb)
            except Exception:
                continue

        result = round(sum(pb_values) / len(pb_values), 2) if pb_values else 0.0
        logger.info(f"{sector} P/B benchmark: ", result)

        bench_data["values"][sector] = result
        bench_data["dates"][sector] = date.today().isoformat()
        self._save_json(bench_path, bench_data)

        return result

    def _current_period_start(self, interval: str) -> datetime:
        """Returns the start of the current incomplete candle in naive UTC."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        threshold = self.STALE_THRESHOLD.get(interval)
        if threshold is None:
            raise ValueError(f"Unknown interval '{interval}'")
        
        # Floor now to the nearest interval
        total_seconds = int(threshold.total_seconds())
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = int((now - epoch).total_seconds())
        floored_seconds = (seconds_since_epoch // total_seconds) * total_seconds
        return epoch + timedelta(seconds=floored_seconds)

    def _is_stale(self, df: pd.DataFrame, interval: str) -> bool:
        if df.empty:
            return True
        
        threshold = self.STALE_THRESHOLD.get(interval)
        if threshold is None:
            raise ValueError(f"Unknown interval '{interval}'")
    
        normalized = self._normalize_tz(df)
        last = normalized.index.max()
        next_candle = last + 2*threshold # 2 because of the open/close difference between two candles
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        is_stale = next_candle < now_utc
        if not is_stale:
            self._weekend_sync_done = False # Reset when data is fresh
            return False
        if self.category == 'stocks' and not BDay().is_on_offset(datetime.now()):
            if self._weekend_sync_done:
                return False  # Already synced once this weekend
            self._weekend_sync_done = True
            return True  # Allow one download
        return True

    ### I/O

    def _drop_from_csv(self, ticker: str, interval: str) -> None:
        path = self._get_price_path(interval)
        if not path:
            logger.error(f"Unknown interval '{interval}'")
            return
        df = self._load_csv(path)
        if ticker in df.columns:
            df = df.drop(columns=[ticker])
            self._save_csv(df, interval)
            # Also evict from memory cache
            if interval in self._hist_prices:
                self._hist_prices[interval] = df
            logger.info(f"{ticker} removed from price history ({interval})")

    def _drop_from_json(self, ticker: str) -> None:
        self._static_metrics = self._load_json(self._metrics_path, default={})
        if ticker in self._static_metrics:
            del self._static_metrics[ticker]
            self._save_json(self._metrics_path, self._static_metrics)
            logger.info(f"{ticker} removed from metrics")

    def _load_csv(self, path) -> pd.DataFrame:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, index_col="Datetime", parse_dates=True).dropna()
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

    def _save_csv(self, df: pd.DataFrame, interval: str):
        path = self._get_price_path(interval)
        df.index.name = "Datetime"
        df.to_csv(path, index=True)

    def _load_json(self, path: str, default: dict) -> dict:
        """Load a JSON file from disk. Returns default if missing, empty, or corrupt."""
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read {path}: {e}")
        return default

    def _save_json(self, path: str, data: dict) -> None:
        """Save a dict to a JSON file. Silently logs on failure."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            logger.error(f"Could not write {path}: {e}")


###
def calculate_growth_rate(divs_filtered):
    """Calculates dividend growth rate (CAGR) using log-linear regression."""
    
    yearly_divs = divs_filtered.groupby(divs_filtered.index.year).sum()
    yearly_divs = yearly_divs[yearly_divs > 0] # Filter for log calculation
            
    growth_rate = "N/A"
    if len(yearly_divs) >= 2:
        # x-values (years from the start)
        x = yearly_divs.index - yearly_divs.index[0]
        
        # y-values (natural log of dividends)
        y = np.log(yearly_divs.values)
                
        # Log-linear regression: slope is compounded growth rate
        #slope, intercept, rvalue, _, _ = linregress(x, y) # Uncomment to visualise regression
        slope, _, _, _, _ = linregress(x, y)
        
        growth_rate = np.exp(slope) - 1
        growth_rate = round(growth_rate, 4)
                
    return growth_rate

        
if __name__ == '__main__':
    # This block is for testing the data fetching function independently
    TICKERS = [
            "SAN.PA", "TT"
        ]
    manager = FinanceDataManager(cache_dir='test', category_name='stocks')
    metrics = manager.get_metrics(TICKERS, interval="15m", force=True)
    for m in metrics:
        print(m)
