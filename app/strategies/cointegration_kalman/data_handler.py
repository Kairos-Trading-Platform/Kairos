from config import Config
import pandas as pd
import numpy as np
import logging # TODO logger
from arch.unitroot import ADF
from statsmodels.tools import add_constant

class DataHandler:
    """
    Price preparation for training and test data.
    Supports rolling windows that can extend into the training set for ADF tests.
    """

    def __init__(self, cfg: Config, df: pd.DataFrame, df2: pd.DataFrame = None, log: bool = True):
        self.cfg = cfg
        self.log = log
        self.df = self._prepare(df)
        self.df_train = self.df  # Alias for clarity

        if df2 is not None:
            self.df_test = self._prepare(df2)
            self.df = pd.concat([self.df_train, self.df_test])  # Combined DataFrame for rolling windows
            self.train_test_boundary = len(self.df_train)  # Index where training ends and test begins
        else:
            self.df_test = None
            self.train_test_boundary = None  # No boundary if only df is provided

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure datetime index and drop initial NaNs
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex(df.index)
        
        # Keep the last duplicated observation for any given timestamp
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]
        # Apply Log transformation if requested
        if self.log:
            # Replacing 0 with epsilon to avoid -inf before log
            df = np.log(df.replace(0, np.nan))

        df = df.replace([np.inf, -np.inf], np.nan).dropna() 
        if self.cfg.check_stationarity:
            self._check_integration_order(df=df)
        return df

    def _check_integration_order(self, df: pd.DataFrame, alpha: float = 0.05):
        for col in df.columns:
            logging.debug(f"Integration check for {col}")
            adf_level = ADF(df[col], trend=self.cfg.adf_trend, method="bic")
            if adf_level.pvalue < alpha:
                raise ValueError(
                    f"Column '{col}' appears stationary in levels "
                    f"(ADF p={adf_level.pvalue:.4f} < {alpha}). "
                    f"Expected I(1) series. Check your input data."
                )
            else:
                logging.info(f"Cannot reject level stationarity hypothesis: p={adf_level.pvalue:.4f} > {alpha})")
            adf_diff = ADF(df[col].diff().dropna(), trend=self.cfg.adf_trend, method="bic")
            if adf_diff.pvalue >= alpha:
                raise ValueError(
                    f"Column '{col}' does not become stationary after one difference "
                    f"(ADF p={adf_diff.pvalue:.4f} >= {alpha}). "
                    f"May be I(2) or have structural breaks. Check your input data."
                )
            else:
                logging.info(f"We reject the first diff stationarity hypothesis: p={adf_diff.pvalue:.4f} < {alpha})")

    def get_window(self, end_idx: int, window: int) -> pd.DataFrame:
        """
        Returns a slice of the prepared data for rolling model fitting.
        If df2 was provided, ensures the window does not extend into the test set.
        """
        start_idx = end_idx - window
        if start_idx < 0:
            # Not enough data yet to fill the window
            return self.df.iloc[0:end_idx] 
        return self.df.iloc[start_idx:end_idx]
        
    def get_observation(self, dep_col: str, indep_cols: list, idx: int = -1):
        """
        Returns a single (y_t, X_t) observation from the prepared data.
        X_t has the constant appended at the end: [x1, x2, ..., 1.0].

        """
        row = self.df.iloc[idx]
        y_t = float(row[dep_col])
        raw = row[indep_cols].values.astype(float).reshape(1, -1)
        X_t = add_constant(raw, has_constant='add', prepend=False).flatten()
        return y_t, X_t

    def summary(self):
        """Print summary."""
        num = len(self.df)
        dates = self.df.index
        print("\nHead:\n", self.df.head(5))
        print("\nTail\n", self.df.tail(5))
        print(f"\nLoaded {num} observations [{dates[0].date()} → {dates[-1].date()}]\n")
        print(self.df.describe().round(2))
