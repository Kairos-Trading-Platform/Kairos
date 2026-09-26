import pandas as pd
from app.utils.finance_data import FinanceDataManager

class PortfolioDataManager:
    def __init__(self, finance: FinanceDataManager):
        self.finance = finance

    def get_live_tickers(self) -> list[str]:
        """Single source of truth for which tickers are in the live portfolio."""
        interval = self.finance.config.get("live_interval")
        path = self.finance._get_price_path(interval)
        df = self.finance._load_csv(path)
        if not df.empty:
            return list(df.columns)
        # Fallback — read tickers from metrics JSON
        metrics = self.finance._load_json(self.finance._metrics_path, default={})
        return list(metrics.keys())

    def get_live_data(self) -> pd.DataFrame:
        tickers = self.get_live_tickers()
        interval = self.finance.config.get("live_interval")
        return self.finance._ensure_prices(tickers, interval, force=False)

class ResearchDataManager:
    def __init__(self, finance_managers: dict, config):
        self.finance_managers = finance_managers
        self.config = config
        # Mirror from live portfolio using stocks manager
        self._portfolio_dm = PortfolioDataManager(finance_managers['stocks'])

    def get_data(self, asset_type: str = 'stocks') -> pd.DataFrame:
        tickers = self._portfolio_dm.get_live_tickers()
        interval = self.config.get("research_interval")
        finance = self.finance_managers[asset_type]
        finance._ensure_prices(tickers, interval, force=False)
        
        df = finance._hist_prices.get(interval)
        if df is None:
            raise RuntimeError(
                f"No data available for interval '{interval}'. "
                f"Available intervals: {list(finance._hist_prices.keys())}. "
                f"Ensure config['research_interval'] matches one of these."
            )
        
        # Verify tickers are columns, not index
        if set(tickers).issubset(df.index):
            df = df.T  # Transpose if tickers are row labels
            df = df.loc[tickers]  # Reorder to match tickers
        
        return df

    def _ticker_asset_type(self, ticker: str) -> str:
        """Resolve which finance manager's universe a ticker belongs to."""
        for asset_type, finance in self.finance_managers.items():
            metrics = finance._load_json(finance._metrics_path, default={})
            if ticker in metrics:
                return asset_type
        raise KeyError(f"Ticker '{ticker}' not found in any configured universe")

    def get_data_for_tickers(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch and merge price history for tickers spanning multiple asset classes."""
        interval = self.config.get("research_interval")
        by_asset_type: dict[str, list[str]] = {}
        for t in tickers:
            by_asset_type.setdefault(self._ticker_asset_type(t), []).append(t)

        frames = []
        for asset_type, group in by_asset_type.items():
            finance = self.finance_managers[asset_type]
            finance._ensure_prices(group, interval, force=False)
            df = finance._hist_prices.get(interval)
            if df is None:
                raise RuntimeError(f"No data for interval '{interval}' in '{asset_type}'")
            if set(group).issubset(df.index):
                df = df.T
            frames.append(df[group])

        return pd.concat(frames, axis=1)