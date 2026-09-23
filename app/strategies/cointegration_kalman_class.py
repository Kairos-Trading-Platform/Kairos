import numpy as np
import os
import re
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
from arch.unitroot import ADF
from arch.unitroot.cointegration import phillips_ouliaris
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from statsmodels.tools import add_constant
from scipy.stats import norm, t as student_t
from scipy.stats import chi2
from itertools import combinations
import joblib
import json
from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from typing import List
import warnings
warnings.filterwarnings("ignore")
import logging

class FirstNFilter(logging.Filter):
    """Only passes the first `n` records that share the same message pattern."""
    def __init__(self, n: int = 3, prefix_len: int = 40):
        super().__init__()
        self.n = n
        self.prefix_len = prefix_len
        self._counts: dict = {}

    def filter(self, record: logging.LogRecord) -> bool:
        # Strip digits so "Coint failed at t=110" and "t=111" share the same key
        msg = record.getMessage()[:self.prefix_len]
        # Only normalise index-like patterns (t=123, fold=2) not metric values
        key = re.sub(r'(t=|fold=|idx=)\d+', r'\1#', msg)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key] <= self.n
    
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

_filter = FirstNFilter(n=3)
root = logging.getLogger()
if not root.handlers:
    root.addHandler(logging.StreamHandler())
for handler in root.handlers:
    handler.addFilter(_filter)

@dataclass
class Config:
    # --- Data ---
    asset_class: str = "equity"   # "equity", "forex", or "index"
    price_path:      str   = "stocks_price_history_1d.csv"
    datetime_col:    str   = "Datetime"
    dep_col:         str   = "AMAT"
    indep_cols:      List[str] = field(default_factory=lambda: ["ASML.AS"])
    test_size:       float = 0.30
    log_prices:      bool  = True

    # Tests
    check_stationarity: bool = True
    coint_enforce_i1: bool = True
    coint_enforce_po: bool = True # initial fit
    coint_enforce_johansen: bool = True
    coint_confirm_n:    int   = 2
    coint_confirm_frac: float = 0.60
 
    # --- Output ---
    output_dir:      str   = "output/"

    # Features dictionary
    Z_SCORE: str = "feat_z_score"
    LOG_Q: str = "feat_log_Q"
    DELTA_ALPHA: str = "feat_delta_alpha"
    BETA_VELOCITY: str = "feat_beta_velocity"
    SPREAD_VOL : str = "feat_spread_vol"
    HALF_LIFE : str = "feat_half_life"
    Z_VELOCITY : str = "feat_z_velocity"
    JOHANSEN_TRACE: str = "feat_johansen_trace"
    KOOPMAN_EIG: str = "koopman_lead_eig"

    # Update dictionary
    PNL_RETURN: str = "pnl_return"
    PNL_GROSS: str = "pnl_gross"
    PNL_DOLLAR: str = "pnl_dollar"
    SIGNAL: str = "signal"
    REGIME_WARNING: str = "regime_warning"
    KOOPMAN_BLOCKED: str = "koopman_blocked"
    COST: str = "cost"
    NOTIONAL: str = "notional"
    SHARES_DEP: str = "shares_dep"
    SHARES_INDEP: str = "shares_indep"
    WEIGHTS_DEP: str = "weight_dep"
    WEIGHTS_INDEP: str = "weight_indep"
    ENTRY_PRICE: str = "entry_price"
    EXIT_PRICE: str = "exit_price"
 
    # --- Distribution for MLE ---
    dist_shape:      str   = "gauss"   # "gauss" or "student"
 
    # --- Kalman filter ---
    kalman_burn_in_raw: int = 100   # user-facing init parameter
    kalman_norm_window: int = 252      # rolling window for the z-score
    kalman_C_prior:  float = 0.01
    kalman_W_default_beta:  float = 1e-1
    kalman_W_default_alpha: float = 1e-3
    kalman_V_default:       float = 1.0
    kalman_debug: bool = False
 
    # --- MLE bounds ---
    mle_W_beta_lo:   float = 1e-10
    mle_W_beta_hi:   float = 1e-3   # tightened from 1e-1: caps daily beta drift at ±3%
    mle_W_alpha_lo:  float = 1e-10
    mle_W_alpha_hi:  float = 1e-1
    mle_V_lo:        float = 1e-4   # fallback floor when fix_V_from_ols=False
    mle_V_hi:        float = 1.0
    mle_n_starts:    int   = 10     # number of random restarts for multi-start MLE
    mle_nu_lo:       float = 2.1
    mle_nu_hi:       float = 50.0
    # When True, V is fixed at the OLS residual variance rather than being
    # optimised.  This resolves the identifiability problem: on short windows
    # the profile likelihood for V is monotonically increasing toward zero,
    # so any unconstrained optimiser will collapse V to its lower bound.
    # OLS residuals give a principled data-grounded estimate of observation
    # noise that is independent of the MLE optimisation.
    mle_fix_V_from_ols: bool = True
 
    # --- Cointegration ---
    coint_maxlags:   int   = 10
    # Minimum number of bars for a cointegration window.
    # Short windows (< 252) give VECM too few mean-reversion cycles to reliably
    # estimate the hedge ratio β and the mean-reversion speed α.  At 100 bars
    # we measured beta errors of 200–600% and half-life errors > 50%.
    # One full trading year (252 bars) is the empirically established minimum
    # for stable VECM estimates.  The scanner will not accept any window shorter
    # than this value, even if it passes the cointegration tests.
    coint_min_window: int  = 252
 
    # --- Signal ---
    entry_z_percentile: float = 90.0  # percentile of abs(z-score) used as entry_z
    exit_z:          float = 0.0
    z_score_std_tol:  float = 0.20
    z_score_std_conf: float = 0.90
    stop_loss_mult:  float = 2.0
    regime_window:   int   = 20
    regime_sigma:    float = 2.0
    regime_filter:   bool  = True
    # Minimum holding period as a fraction of the estimated half-life.
    # Prevents the z-score from triggering an exit before the spread has
    # had time to mean-revert.  Without this, the Kalman filter's dynamic
    # Q causes z_score to cross zero in 2–7 bars even when the true
    # half-life is 10–34 bars, leaving most of the expected profit
    # uncaptured.  The stop-loss and time-decay exits are NOT blocked —
    # only the normal target-reversion exit is deferred.
    # 0.5 means "wait at least half a half-life before allowing normal exit."
    min_hold_frac:   float = 0.5
 
    # --- Backtest ---
    bt_window:       int   = 252
    label_horizon:   int   = 5
 
    # --- Portfolio ---
    capital:         float = 100_000
    target_vol:      float = 0.01
    cost_bps:        float = 5.0
 
    # --- RF ---
    rf_n_estimators: int   = 200
    rf_n_splits:     int   = 5
    rf_threshold:    float = 0.55

    # --- XGBoost ---
    xgb_n_estimators:   int   = 300
    xgb_max_depth:      int   = 4
    xgb_learning_rate:  float = 0.05
    xgb_subsample:      float = 0.8
    xgb_threshold:      float = 0.55

    # --- Koopman ---
    koopman_window:     int   = 60   # obs used for EDMD estimation
    koopman_n_obs:      int   = 10   # size of observable dictionary (delays)
    koopman_eig_thresh: float = 0.97 # leading |eigenvalue| must be < this to allow entry

    # --- Walk-forward ---
    wf_n_splits:        int   = 5    # number of expanding folds
    wf_min_train_frac:  float = 0.40 # minimum fraction of training data in first fold

    # Production
    replay: bool = True
 
    # --- Derived (computed at runtime, not set by user) ---
    state_dim: int = field(init=False)
    z_score_min_bars: int = field(init=False)
    _kalman_burn_in: int = field(init=False, repr=False)
    config_file: str = field(init=False)
    state_file: str = field(init=False)
    qm_file: str = field(init=False)
    hist_file: str = field(init=False)
    bt_file: str = field(init=False)
    filtered_file: str = field(init=False)
    innov_file: str = field(init=False)
    coeff_file: str = field(init=False)
 
    def __post_init__(self):
        self.output_dir = self.output_dir_name()
        self.state_dim  = len(self.indep_cols) + 1   # betas + alpha
        self.config_file = self.output_dir + "config.json"
        self.state_file = self.output_dir + "kalman_state.pkl"
        self.xgb_file   = self.output_dir + "xgb_quality_model.pkl"
        self.qm_file    = self.output_dir + "signal_quality_model.pkl"
        self.hist_file  = self.output_dir + "history.csv"
        self.bt_file    = self.output_dir + "bt_results.csv"
        self.filtered_file = self.output_dir + "filtered_results.csv"
        self.innov_file = self.output_dir + "innovation_distribution.png"
        self.coeff_file = self.output_dir + "cointegration_coefficients.png"
        self.z_score_min_bars = self._get_z_score_min_bar()
        self.kalman_burn_in = self.kalman_burn_in_raw # goes through setter
        logging.debug(f"z_score_min_bars: {self.z_score_min_bars}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Deterministic specifications derived from asset class
        _specs = {
            "equity": {
                "adf_trend":        "c",
                "johansen_det":     0,       # restricted constant
                "vecm_det":         "ci",    # constant in cointegrating relation
                "ols_spread_const": True,    # include constant in half-life OLS
            },
            "forex": {
                "adf_trend":        "n",
                "johansen_det":    -1,       # no deterministic terms
                "vecm_det":         "n",     # no constant anywhere
                "ols_spread_const": False,
            },
            "index": {
                "adf_trend":        "c",
                "johansen_det":     0,
                "vecm_det":         "ci",
                "ols_spread_const": True,
            },
        }
        if self.asset_class not in _specs:
            raise ValueError(
                f"asset_class must be one of {list(_specs.keys())}, "
                f"got '{self.asset_class}'"
            )
        spec = _specs[self.asset_class]
        self.adf_trend          = spec["adf_trend"]
        self.johansen_det       = spec["johansen_det"]
        self.vecm_det           = spec["vecm_det"]
        self.ols_spread_const   = spec["ols_spread_const"]

    @property
    def kalman_burn_in(self) -> int:
        return self._kalman_burn_in

    @kalman_burn_in.setter
    def kalman_burn_in(self, value: int):
        if hasattr(self, 'z_score_min_bars') and value < self.z_score_min_bars:
            logging.warning(
                f"kalman_burn_in={value} < z_score_min_bars="
                f"{self.z_score_min_bars}. "
                f"Using {self.z_score_min_bars} instead."
            )
            self._kalman_burn_in = self.z_score_min_bars
        else:
            self._kalman_burn_in = value

    def _get_z_score_min_bar(self) -> int:
        alpha = 1 - self.z_score_std_conf
        for n in range(2, 10000):
            lo = np.sqrt((n - 1) / chi2.ppf(1 - alpha / 2, df=n - 1))
            hi = np.sqrt((n - 1) / chi2.ppf(    alpha / 2, df=n - 1))
            if (hi - 1) <= self.z_score_std_tol and (1 - lo) <= self.z_score_std_tol:
                return n
        return 200

    def output_dir_name(self) -> str:
        clean = lambda s: s.split(".")[0]   # "ASML.AS" → "ASML"
        dep   = clean(self.dep_col)
        indep = "_".join(clean(col) for col in self.indep_cols)
        return os.path.join(self.output_dir, f"{dep}_vs_{indep}") + os.sep

    def save(self, file_path: str | Path) -> None:
        """Save the Config object to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls, file_path: str | Path) -> "Config":
        """Load a Config object from a JSON file."""
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        # Remove any field that has init=False, __post_init__ will recompute them
        non_init = {f.name for f in dataclasses.fields(cls) if not f.init}
        for key in non_init:
            config_dict.pop(key, None)
        return cls(**config_dict)

@dataclass
class StepResult:
    e: float   # raw innovation
    Q: float   # innovation variance
    m: np.ndarray  # updated state
    z_score: float  
    innov_std: float   # std(buf_e), volatility of raw innovations
    z_std: float  # std(buf_z), volatility of z_score
    C: np.ndarray = None # updated state covariance matrix

# 1.Load data
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

# 2. Load co-integration model
class CointegrationModel:
    """
    Fits a VECM and runs Johansen + ADF diagnostics on a price window.
    Returns a structured result dict consumed by KalmanModel initialisation.
    """

    def __init__(self, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to CointegrationModel.")
        self.cfg = cfg

    def fit(self, df_window: pd.DataFrame) -> dict:
        results = {}
        
        # ADF tests (Level & Diff)
        for col in df_window.columns:
            logging.debug(f"Trend for {col}: {self.cfg.adf_trend}")
            lvl = ADF(df_window[col], trend=self.cfg.adf_trend, method="bic")
            logging.debug(f"lvl: {lvl}")
            df1 = ADF(df_window[col].diff().dropna(), trend=self.cfg.adf_trend, method="bic")
            logging.debug(f"df1: {df1}")
            results[f"ADF_{col}_lvl_stat"] = lvl.stat
            results[f"ADF_{col}_lvl_crit"] = lvl.critical_values['5%']
            results[f"ADF_{col}_dif_stat"] = df1.stat
            results[f"ADF_{col}_dif_crit"] = df1.critical_values['5%']

            if lvl.stat < lvl.critical_values['5%']:
                logging.debug(f"if lvl")
                if self.cfg.coint_enforce_i1:
                    raise ValueError(
                        f"'{col}' appears I(0) in this window "
                        f"(ADF stat {lvl.stat:.3f} < crit {lvl.critical_values['5%']:.3f}). "
                        f"VECM requires I(1) inputs."
                    )
                else:
                    logging.warning(
                        f"'{col}' appears I(0) in this window, continuing because "
                        f"coint_enforce_i1=False."
                    )
            if df1.stat > df1.critical_values['5%']:        # fails to reject after differencing -> I(2)+
                logging.warning(
                    f"'{col}' may be I(2) in this window (ADF on diff: "
                    f"{df1.stat:.3f} > crit {df1.critical_values['5%']:.3f}). Proceeding with caution."
                )

        # Phillips Ouliaris test
        po = phillips_ouliaris(
                            df_window[self.cfg.dep_col], df_window[self.cfg.indep_cols], trend=self.cfg.adf_trend, test_type="Za", kernel="bartlett"
                        )
        results["PO_stat"] = po.stat
        results["PO_crit"] = po.critical_values[5]
        results["PO_p_value"] = po.pvalue

        if po.stat > po.critical_values[5]:
            if self.cfg.coint_enforce_po:
                raise ValueError(
                    f"PO t-stat {po.stat:.3f} > crit {po.critical_values[5]:.3f}). "
                    f"Pairs may not be co-integrated."
                )
            else:
                logging.debug(
                    f"PO test failed (stat {po.stat:.3f} > crit "
                    f"{po.critical_values[5]:.3f}) — continuing because "
                    f"coint_enforce_po=False."
                )

        # VAR selection
        k_ar_diff = VAR(df_window).select_order(self.cfg.coint_maxlags).bic
        results["VAR_lag_bic"] = k_ar_diff
        logging.debug(f"VAR selection: {k_ar_diff}")

        # Johansen test
        jtest = coint_johansen(df_window, det_order=self.cfg.johansen_det, k_ar_diff=k_ar_diff)
        results["johansen"] = jtest
        logging.debug(f"Johansen test: {jtest}")

        # Dynamically determine the number of ranks to test
        num_cols = len(df_window.columns)
        results["Johansen_r=0_stat"] = jtest.trace_stat[0]
        results[f"Johansen_r=0_crit_95"] = jtest.trace_stat_crit_vals[0, 1]
        for i in range(1, num_cols):
            results[f"Johansen_r<={i}_stat"] = jtest.trace_stat[i]
            results[f"Johansen_r<={i}_crit_95"] = jtest.trace_stat_crit_vals[i, 1]

        if results["Johansen_r=0_stat"] <= results["Johansen_r=0_crit_95"]:
            if self.cfg.coint_enforce_johansen:
                raise ValueError(
                    f"Johansen trace test fails to reject r=0 "
                    f"(stat={results['Johansen_r=0_stat']:.3f} <= "
                    f"crit={results['Johansen_r=0_crit_95']:.3f}). "
                    f"No evidence of cointegration at the 95% level."
                )
            else:
                logging.warning(
                    f"Johansen trace test fails to reject r=0 — proceeding "
                    f"because coint_enforce_johansen=False."
                )

        # VECM coefficients
        vecm_res = VECM(df_window, k_ar_diff=k_ar_diff, deterministic=self.cfg.vecm_det).fit()
        results["vecm"] = vecm_res
        raw_beta = vecm_res.beta[:, 0]
        if self.cfg.vecm_det == "n":
            raw_alpha = 0.0
        else:
            raw_alpha = float(vecm_res.const_coint.flat[0])
        results["VECM_beta"] = raw_beta # Cointegrating vector: beta is normalised so first variable has coeff 1.
        results["VECM_alpha"] = raw_alpha

        # Residual (Spread) stationarity
        spread = df_window.values @ raw_beta + raw_alpha
        logging.debug(f"Spread stationarity: {spread}")

        #Half-Life
        # Spread_t = rho * Spread_{t-1} + e
        # Half-life = -log(2) / log(abs(rho))
        spread_series = pd.Series(spread)
        z_lag = spread_series.shift(1).dropna()
        z_diff = spread_series.diff().dropna()
        # Simple OLS: delta_z = alpha + beta * z_{t-1}
        # Discrete version of the OU process
        if self.cfg.ols_spread_const:
            reg = sm.OLS(z_diff, sm.add_constant(z_lag)).fit()
            lambda_val = reg.params.iloc[1]
        else:
            reg = sm.OLS(z_diff, z_lag).fit()
            lambda_val = reg.params.iloc[0]
        if lambda_val < 0:
            results["half_life"] = -np.log(2) / lambda_val
        else:
            results["half_life"] = 1000.0  # Cap it if it's not reverting

        spread_adf = ADF(spread, trend="n", method="bic") # Trend is none after VECM fitting
        results["Spread_ADF_stat"] = spread_adf.stat
        results["Spread_ADF_crit_95"] = spread_adf.critical_values['5%']
        results["Spread_pvalue"] = spread_adf.pvalue

        logging.debug(f"Spread ADF: {spread_adf}")
        return results
        
    def get_model_confidence(self, results: dict) -> dict:
        """
        Normalises test results into a 0-1 range.
        Values > 1.0 indicate the test passed.
        Values < 1.0 indicate how close it was to passing.
        """
        conf = {}
        
        # Johansen strength (ratio of stat to 95% crit val)
        # If > 1, the null of 'no cointegration' is rejected.
        conf['johansen_strength'] = results["Johansen_r=0_stat"] / results["Johansen_r=0_crit_95"]
        
        # Residual stationarity strength
        # We use (1 - p-value) so that 1.0 is perfectly stationary and 0.0 is a unit root.
        conf['resid_stationarity'] = 1 - results["Spread_pvalue"]
        
        # Distance from critical value (ADF)
        # How many times larger is our test stat than the critical value?
        conf['resid_adf_ratio'] = abs(results["Spread_ADF_stat"]) / abs(results["Spread_ADF_crit_95"])

        # Total probability score
        weights = {'johansen_strength': 0.4, 'resid_stationarity': 0.6}
        
        # Cap strengths at 1.2 to avoid outliers skewing the ML
        norm_j = min(conf['johansen_strength'], 1.2)
        norm_s = min(conf['resid_stationarity'], 1.0)
        
        conf['total_confidence'] = (norm_j * weights['johansen_strength']) + (norm_s * weights['resid_stationarity'])
        
        return conf
    
    def check_health(self, df_window: pd.DataFrame, beta: np.ndarray, alpha: float) -> float:
        """
        Returns the p-value of the ADF test on the spread 
        calculated with CURRENT model parameters.
        """
        # Calculate the spread using current Kalman weights: spread = Y - (X * beta)
        # If your beta includes the intercept, ensure X has the constant.
        # Otherwise: spread = Y - (X @ beta_cols + alpha)
        
        # Assuming df_window columns match your model order
        y = df_window[self.cfg.dep_col].values
        X = df_window[self.cfg.indep_cols].values
        
        # Calculate spread: Y_t - (beta1*X1 + beta2*X2 + ... + alpha)
        spread = y - (X @ beta + alpha)
        
        # Run ADF on the residual. We keep trend="n" to detect regime break
        try:
            res = ADF(spread, trend="n", method="bic")
            return res.pvalue
        except:
            return 1.0 # Return failure if test crashes
        
# 3. Load Kalman filter
class KalmanModel:
    """
    Linear Gaussian state-space filter tracking time-varying cointegration
    coefficients.
 
    State:    m_t  = [beta1_t, beta2_t, ..., alpha_t]
    Obs eq:   y_t  = X_t . m_t + eps_t,   eps ~ N(0, V)
    State eq: m_t  = m_{t-1} + eta_t,     eta ~ N(0, W)
    """

    def __init__(self, W_diag=None, V=1.0, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to KalmanModel.")
        self.cfg = cfg
        # W represents how much we allow coefficients to drift per step
        if W_diag is None:
            self.W = np.diag([self.cfg.kalman_W_default_beta] * (self.cfg.state_dim - 1) + [self.cfg.kalman_W_default_alpha])
        else:
            if isinstance(W_diag, list):
                W_diag = np.array(W_diag)
            if W_diag.ndim == 1:
                self.W = np.diag(W_diag)
            else:
                self.W = W_diag

        self.V = V
        self.m = None
        self.C = None
        self.nu = 5.0
        self._innov_history  = []   # rolling buffer for past e values
        self._z_score_history = []  # rolling buffer for past z_score values
        self.last_date = None

        # Bind the step function dynamically based on config
        if self.cfg.kalman_debug:
            self.step = self._step_debug
        else:
            self.step = self._step_production

    def initialise(self, beta_indep: np.ndarray, alpha: float):
        """
        beta_indep : coefficients for the independent variables only
                     (i.e. VECM_beta[1:], already excluding the dep-var coeff)
        alpha      : VECM intercept (sign unchanged from CointegrationModel)
        """

        self.m = np.array([*beta_indep, alpha], dtype=float)
        self.C = np.eye(self.cfg.state_dim) * self.cfg.kalman_C_prior

    def step(self, y_t: float, X_t: np.ndarray) -> StepResult:
        """This will be dynamically overwritten in __init__"""
        pass

    def _step_debug(self, y_t: float, X_t: np.ndarray) -> StepResult:
        """
        y_t : scalar observation (log-price of dependent variable)
        X_t : 1-D array [log(x1), log(x2), ..., 1.0]
 
        Returns (innovation e, forecast variance Q, updated state m)
        """
        # Prediction
        # a_t = m_{t-1}, R_t = C_{t-1} + W
        R = self.C + self.W # predicted covariance

        # Forecast
        f = X_t @ self.m # predicted observation
        Q = X_t @ R @ X_t.T + self.V # innovation variance

        # Update
        e = y_t - f # innovation
        K = (R @ X_t.T) / Q # Kalman gain

        self.m = self.m + K * e
        self.C = (np.eye(self.cfg.state_dim) - np.outer(K, X_t)) @ R

        # update buffer
        z_model = e / np.sqrt(Q)
        self._innov_history.append(e)
        self._z_score_history.append(z_model)
        buf_e = self._innov_history[-self.cfg.kalman_norm_window:]
        buf_z = self._z_score_history[-self.cfg.kalman_norm_window:]
        sigma=0
        z_std=0
        if len(buf_e) >= self.cfg.z_score_min_bars:
            sigma   = np.std(buf_e)
            z_std   = np.std(buf_z)
            z_score = (z_model - np.mean(buf_z)) / z_std if z_std > 1e-10 else np.nan
        else:
            z_score = np.nan

        return StepResult(e=e, Q=Q, m=self.m.copy(), C=self.C.copy(),
                          z_score=z_score,
                          innov_std=sigma, z_std=z_std)
    
    def _step_production(self, y_t: float, X_t: np.ndarray) -> StepResult:
        """
        y_t : scalar observation (log-price of dependent variable)
        X_t : 1-D array [log(x1), log(x2), ..., 1.0]
 
        Returns (innovation e, forecast variance Q, updated state m)
        """
        # Prediction
        # a_t = m_{t-1}, R_t = C_{t-1} + W
        R = self.C + self.W # predicted covariance

        # Forecast
        f = X_t @ self.m # predicted observation
        Q = X_t @ R @ X_t.T + self.V # innovation variance

        # Update
        e = y_t - f # innovation
        K = (R @ X_t.T) / Q # Kalman gain

        self.m = self.m + K * e
        self.C = (np.eye(self.cfg.state_dim) - np.outer(K, X_t)) @ R

        # update buffer
        z_model = e / np.sqrt(Q)
        self._innov_history.append(e)
        self._z_score_history.append(z_model)
        buf_e = self._innov_history[-self.cfg.kalman_norm_window:]
        buf_z = self._z_score_history[-self.cfg.kalman_norm_window:]
        sigma=0
        z_std=0
        if len(buf_e) >= self.cfg.z_score_min_bars:
            sigma   = np.std(buf_e)
            z_std   = np.std(buf_z)
            z_score = (z_model - np.mean(buf_z)) / z_std if z_std > 1e-10 else np.nan
        else:
            z_score = np.nan

        return StepResult(e=e, Q=Q, m=self.m.copy(),
                          z_score=z_score,
                          innov_std=sigma, z_std=z_std)
    
    def run_and_report(self, y: np.ndarray, X: np.ndarray,
                       burn_in: int = None, nu: float = None,
                       labels: list = None, dates=None) -> pd.DataFrame:
        """
        Runs the filter over the full training series, prints diagnostics,
        and returns a tidy DataFrame (post burn-in).
        """
        results = [self.step(y[t], X[t]) for t in range(len(y))]
        
        if burn_in is None:
            burn_in = self.kalman_burn_in   # property getter — already enforced
        else:
            # Enforce minimum even when passed explicitly
            burn_in = max(burn_in, self.cfg.z_score_min_bars)
    
        # Unpack post burn-in
        e_s          = np.array([r.e for r in results[burn_in:]])
        Q_s          = np.array([r.Q for r in results[burn_in:]])
        z_score     = np.array([r.z_score for r in results[burn_in:]])
        m_s          = np.array([r.m for r in results[burn_in:]])
        idx          = (dates[burn_in:] if dates is not None
                        else np.arange(len(e_s)))
        
        # Remove any NaN/Inf rows uniformly
        valid = np.isfinite(z_score)
        e_s, Q_s, m_s = e_s[valid], Q_s[valid], m_s[valid]
        z_score_s = z_score[valid]
        if hasattr(idx, '__getitem__'):
            idx = idx[valid]

        # Diagnostics
        self._print_diagnostics(z_score_s, nu)
        self._plot_results(z_score_s, nu, m_s, idx, labels)

        columns = {
            'z_score': z_score_s,
            'spread': e_s,
            'variance_Q': Q_s,
        }

        # Add state coefficients (beta_1, beta_2, ..., alpha)
        for i in range(self.cfg.state_dim):
            if i < self.cfg.state_dim - 1:
                columns[f'beta_{i+1}'] = m_s[:, i]
            else:
                columns['alpha'] = m_s[:, i]

        return pd.DataFrame(columns, index=idx)

    def _print_diagnostics(self, z, nu):
        print(f"\n--- Z-Score ---")
        logging.info(f"Mean: {z.mean():.5f} | Std: {z.std():.5f}")
        logging.info(f"Skew: {stats.skew(z):.5f} | Kurt: {stats.kurtosis(z):.5f}")
        jb_p = stats.jarque_bera(z).pvalue
        sw_p = stats.shapiro(z[:5000]).pvalue
        logging.info(f"JB p: {jb_p:.5f} | SW p: {sw_p:.5f}")
        logging.info(f"Min/Max: {z.min():.4f} / {z.max():.4f}")
        if nu:
            logging.info(f"Student-t 95% threshold (nu={nu:.2f}): "
                f"{student_t.ppf(0.975, df=nu):.4f}")

    def _plot_results(self, z_score, nu, m_s, idx, labels):
        # Distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(z_score, bins=50, density=True, alpha=0.6, label=f'Z-score')
        x = np.linspace(z_score.min(), z_score.max(), 100)
        plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label='Normal Dist')
        if nu:
            plt.plot(x, stats.t.pdf(x, df=nu), 'b-', lw=2, label=f"Student's t (nu={nu:.2f})")
        plt.title("Innovation distribution")
        plt.legend()
        plt.savefig(self.cfg.innov_file)
        plt.close()

        # Coefficient evolution plot
        lbl = labels or [f'State {i}' for i in range(self.cfg.state_dim)]
        fig, axes = plt.subplots(self.cfg.state_dim, 1, figsize=(12, 4 * self.cfg.state_dim), sharex=True)
        for i in range(self.cfg.state_dim):
            axes[i].plot(idx, m_s[:, i], alpha=0.8)
            axes[i].set_title(f'Evolution of {lbl[i]}')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.cfg.coeff_file)
        plt.close()

    def save_state(self, filepath="kalman_state.pkl"):
        joblib.dump({
            'm':         self.m,
            'C':         self.C,
            'W':         self.W,
            'V':         self.V,
            'nu':        self.nu,
            'last_date': getattr(self, 'last_date', None),
            'innov_history':   self._innov_history,
            'z_score_history': self._z_score_history,
            'entry_z': getattr(self, 'entry_z', 1.0),
            'exit_z':  getattr(self, 'exit_z',  0.0),
        }, filepath)


    @classmethod
    def load_state(cls, cfg: Config) -> 'KalmanModel':
        data     = joblib.load(cfg.state_file)
        instance = cls(W_diag=np.diag(data['W']), V=data['V'], cfg=cfg)
        instance.m         = data['m']
        instance.C         = data['C']
        instance.nu        = data.get('nu', 5.0)
        instance.last_date = data.get('last_date')
        instance._innov_history   = data.get('innov_history',   [])
        instance._z_score_history = data.get('z_score_history', [])
        instance.entry_z = data.get('entry_z', 1.0)
        instance.exit_z  = data.get('exit_z',  0.0)
        return instance

class KalmanMLE:
    """
    Optimises noise hyperparameters (W_beta, W_alpha, V, nu) by maximising
    the log-likelihood of the Kalman innovations.
    """

    def __init__(self, init_beta: np.ndarray,
                 init_alpha: float, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to KalmanMLE.")
        self.cfg = cfg
        self.init_beta = init_beta
        self.init_alpha = init_alpha
        

    def _neg_ll(self, log_params, y, X):
        params = np.exp(log_params)

        if self.cfg.dist_shape == "student":
            expected_params = 3 + (self.cfg.state_dim - 1)  # W_beta (state_dim-1), W_alpha, V, df
            if len(params) != expected_params:
                raise ValueError(f"Student-t requires {expected_params} parameters: W_beta_1..{self.cfg.state_dim-1}, W_alpha, V, df")
            W_beta = params[:self.cfg.state_dim - 1]  # First state_dim-1 parameters
            W_alpha, Vk, df = params[self.cfg.state_dim - 1], params[self.cfg.state_dim], params[self.cfg.state_dim + 1]

        elif self.cfg.dist_shape == "gauss":
            expected_params = 2 + (self.cfg.state_dim - 1)  # W_beta (state_dim-1), W_alpha, V
            if len(params) != expected_params:
                raise ValueError(f"Gaussian requires {expected_params} parameters: W_beta_1..{self.cfg.state_dim-1}, W_alpha, V")
            W_beta = params[:self.cfg.state_dim - 1]  # First state_dim-1 parameters
            W_alpha, Vk = params[self.cfg.state_dim - 1], params[self.cfg.state_dim]
            df = None
        else:
            raise ValueError("dist must be 'gauss' or 'student'")
    
        W_diag = np.append(W_beta, W_alpha)
        kf = KalmanModel(W_diag=W_diag,V=Vk, cfg=self.cfg)
        kf.initialise(self.init_beta, self.init_alpha)
        ll = 0.0
        for t in range(len(y)):
            sr = kf.step(y[t], X[t])
            scale = np.sqrt(sr.Q)
            if self.cfg.dist_shape == "student":
                ll += student_t.logpdf(sr.e, df=df, loc=0, scale=scale)
            else:  # gaussian
                ll += norm.logpdf(sr.e, loc=0, scale=scale)
        return -ll

    @staticmethod
    def _v_from_ols(y: np.ndarray, X: np.ndarray) -> float:
        """
        Estimate observation noise V from OLS residual variance.

        Why this works: fitting y = X @ beta_ols + residual with a fixed beta
        gives residuals that capture the full spread noise (OU process realisations
        plus any beta-drift contribution).  Their variance is therefore an upper
        bound on V — the true V can be somewhat smaller because some spread
        variance comes from beta drift — but it is always in the right order of
        magnitude and will never collapse to zero.

        We use the full OLS variance (no shrinkage factor) because:
          1. On short windows the OLS beta estimate absorbs some true noise,
             which slightly under-estimates the residual variance.
          2. A conservative (slightly large) V causes the Kalman gain to be
             slightly smaller, which is the safer direction — it makes the
             filter more stable, not less.
        """
        try:
            beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
            resid    = y - X @ beta_ols
            return float(np.var(resid))
        except Exception:
            return None

    def fit(self, y: np.ndarray, X: np.ndarray):
        """
        Maximise the Kalman log-likelihood over W_beta and W_alpha using
        multi-start L-BFGS-B.

        V handling (mle_fix_V_from_ols=True, the default):
            V is fixed at the OLS residual variance rather than being
            optimised.  Profile-likelihood analysis showed that the likelihood
            for V is monotonically increasing as V → 0 on windows of 100–200
            bars: there is no interior optimum and any unconstrained optimiser
            will collapse V to its lower bound regardless of where that bound
            is set.  This is a fundamental identifiability problem — with short
            data, W and V cannot be separately estimated from the innovations
            alone.  Fixing V from OLS breaks the degeneracy: OLS residual
            variance is a principled, data-grounded estimate that reflects the
            true scale of observation noise without relying on the optimiser.

        V handling (mle_fix_V_from_ols=False):
            V is included in the optimisation with multi-start L-BFGS-B.
            Useful for long windows (>500 bars) where V and W become
            separately identifiable, or for diagnostic comparisons.
        """
        fix_V    = self.cfg.mle_fix_V_from_ols
        V_fixed  = None

        if fix_V:
            V_fixed = self._v_from_ols(y, X)
            if V_fixed is None or V_fixed <= 0:
                logging.warning(
                    "OLS V estimation failed — falling back to optimising V.")
                fix_V   = False
                V_fixed = None
            else:
                # Clamp to the configured feasible region so Config bounds
                # are still respected even in fixed-V mode.
                V_fixed = float(np.clip(V_fixed, self.cfg.mle_V_lo,
                                        self.cfg.mle_V_hi))
                logging.info(f"V fixed from OLS residuals: {V_fixed:.4e}")

        # Build bounds over W parameters only (V excluded when fix_V=True)
        bounds = []
        for _ in range(self.cfg.state_dim - 1):
            bounds.append((np.log(self.cfg.mle_W_beta_lo),
                           np.log(self.cfg.mle_W_beta_hi)))
        bounds.append((np.log(self.cfg.mle_W_alpha_lo),
                       np.log(self.cfg.mle_W_alpha_hi)))
        if not fix_V:
            bounds.append((np.log(self.cfg.mle_V_lo),
                           np.log(self.cfg.mle_V_hi)))
        if self.cfg.dist_shape == 'student':
            bounds.append((np.log(self.cfg.mle_nu_lo),
                           np.log(self.cfg.mle_nu_hi)))

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Wrapper: if V is fixed, inject it before evaluating the likelihood
        if fix_V:
            def _objective(log_w_params, y, X):
                # Reconstruct full param vector: [...W_params..., V_fixed, ...]
                n_w = self.cfg.state_dim          # W_beta_1...W_alpha
                if self.cfg.dist_shape == 'student':
                    full = np.concatenate([log_w_params[:n_w],
                                           [np.log(V_fixed)],
                                           log_w_params[n_w:]])
                else:
                    full = np.concatenate([log_w_params, [np.log(V_fixed)]])
                return self._neg_ll(full, y, X)
        else:
            def _objective(log_params, y, X):
                return self._neg_ll(log_params, y, X)

        # Default starting point (physically motivated)
        n_w = self.cfg.state_dim
        w_defaults = [1e-5] * (n_w - 1) + [1e-4]   # W_beta..W_alpha
        if fix_V:
            x0_base = np.log(w_defaults)
            if self.cfg.dist_shape == 'student':
                x0_base = np.append(x0_base, np.log(5.0))
        else:
            x0_base = np.log(w_defaults + [1e-3])   # include V
            if self.cfg.dist_shape == 'student':
                x0_base = np.append(x0_base, np.log(5.0))

        best_ll = np.inf
        best_x  = x0_base.copy()
        n_starts = max(1, self.cfg.mle_n_starts)

        for start_idx in range(n_starts):
            x0 = x0_base if start_idx == 0 else np.random.uniform(lo, hi)
            try:
                res = minimize(_objective, x0=x0, args=(y, X),
                               bounds=bounds, method='L-BFGS-B')
                if res.fun < best_ll:
                    best_ll = res.fun
                    best_x  = res.x
            except Exception as exc:
                logging.warning(f"MLE start {start_idx+1}/{n_starts} failed: {exc}")

        if best_ll == np.inf:
            logging.warning("All MLE starts failed — returning defaults.")
            best_x = x0_base

        # Reconstruct full opt array: [W_beta_1,...,W_alpha, V, (nu)]
        W_opt = np.exp(best_x[:n_w])
        V_out = V_fixed if fix_V else float(np.exp(best_x[n_w]))

        if not fix_V and V_out <= self.cfg.mle_V_lo * 1.01:
            logging.warning(
                f"MLE V = {V_out:.2e} is at its lower bound "
                f"({self.cfg.mle_V_lo:.2e}). The profile likelihood for V "
                "is monotonically increasing toward zero on this window. "
                "Set mle_fix_V_from_ols=True to resolve this.")

        opt = np.concatenate([W_opt, [V_out]])
        if self.cfg.dist_shape == 'student':
            nu_out = float(np.exp(best_x[-1]))
            opt = np.append(opt, nu_out)

        mode = "V fixed from OLS" if fix_V else f"{n_starts} starts"
        msg = (f"MLE ({mode}, ll={-best_ll:.4f})  "
               f"W_beta={' '.join(f'{W_opt[i]:.2e}' for i in range(n_w-1))}  "
               f"W_alpha={W_opt[-1]:.2e}  V={V_out:.2e}")
        if self.cfg.dist_shape == 'student':
            msg += f"  nu={opt[-1]:.2f}"
        logging.info(msg)
        return opt  # [W_beta_1,...,W_alpha, V] or [..., nu] for Student-t

# 4. Analyse statistical tests
class StatisticalTestEngine:
    """Rolling ADF, half-life, and CUSUM diagnostics."""
    def compute_all(self, df: pd.DataFrame, spread: pd.Series,
                    johansen_res) -> dict:
        result = {}
 
        for key, series in [
            ("adf_level_p", df.iloc[:, 0], "n"),
            ("adf_resid_p", spread,         "n"),
        ]:
            s = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 20 or s.std() < 1e-8:
                result[key] = np.nan
            else:
                # Cap max_lags so ADF never requests more lags than the
                # series can support (rule of thumb: n/5, capped at 10)
                safe_lags = min(10, max(1, len(s) // 5))
                try:
                    result[key] = ADF(s, trend="n", method="bic",
                                      max_lags=safe_lags).pvalue
                except Exception as exc:
                    logging.error(f"  ADF failed [{key}, n={len(s)}, "
                          f"max_lags={safe_lags}]: {exc}")
                    result[key] = np.nan
 
        result["johansen_trace"] = johansen_res.trace_stat[0]
        result["half_life"]      = self._estimate_half_life(spread)
        return result

    def _estimate_half_life(self, spread: pd.Series) -> float:
        s = spread.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20:
            return np.nan
        df = pd.DataFrame({"lag": s.shift(1), "ret": s - s.shift(1)}).dropna()
        if len(df) < 20:
            return np.nan
        lam = Polynomial.fit(df["lag"], df["ret"], 1).convert().coef[1]
        return np.nan if lam >= 0 else -np.log(2) / lam


    def cusum_test(self, std_innov: np.ndarray, significance: float = 0.05,
                   burn_in: int = 0):
        n   = len(std_innov)
        eff = std_innov[burn_in:]
        n_e = len(eff)
        c   = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}[significance]
 
        cusum_val = np.cumsum(eff) / np.sqrt(n_e)
        t_i       = np.arange(1, n_e + 1)
        upper_val =  c * np.sqrt(t_i / n_e)
        lower_val = -c * np.sqrt(t_i / n_e)
 
        cusum = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        cusum[burn_in:] = cusum_val
        upper[burn_in:] = upper_val
        lower[burn_in:] = lower_val
 
        detected, in_break = [], False
        for i in range(2, n_e):
            outside = cusum[i] > upper[i] or cusum[i] < lower[i]
            if not in_break and outside:
                detected.append(i + burn_in)
                in_break = True
            elif in_break and not outside:
                in_break = False
 
        return cusum, upper, lower, detected


    def rolling_adf(self, residuals: np.ndarray, window: int = 120):
        num = len(residuals)
        adf_stats, crit_5pct = np.full(num, np.nan), np.full(num, np.nan)

        for t in range(window, num):
            try:
                # trend="n" is correct for residuals
                res = ADF(residuals[t - window : t], trend="n")
                adf_stats[t] = res.stat
                crit_5pct[t] = res.critical_values["5%"]
            except:
                continue
        return adf_stats, None, crit_5pct # keep return signature consistent

# 5. Define signal
class SignalEngine:
    """
    Mean-reversion signal from z-score with hysteresis and regime warning.
    Regime warning uses a rolling baseline of |delta_alpha| so the threshold
    adapts to the filter's typical daily adjustment rather than a fixed value.
    """

    def __init__(self, entry_z: float = 1.0, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to SignalEngine.")
        self.cfg = cfg
        self.entry_z              = entry_z
        self.exit_z               = abs(cfg.exit_z)
        self.regime_window        = cfg.regime_window
        self.regime_sigma         = cfg.regime_sigma
        self.regime_filter        = cfg.regime_filter
        self.stop_loss_multiplier = cfg.stop_loss_mult
        self.min_hold_frac        = cfg.min_hold_frac
        self.position             = 0
        self.bars_in_trade        = 0
        self._alpha_history       = []

    def generate(self, z: float, delta_alpha: float = 0.0, half_life: float = 20.0):
        """
        Returns (position, regime_warning).
 
        regime_warning is True when |delta_alpha| exceeds the rolling mean
        + regime_sigma * rolling_std of recent alpha changes.
        """
        self._alpha_history.append(abs(delta_alpha))
        buf = self._alpha_history[-self.regime_window:]

        # Regime detection
        if len(buf) >= 5:
            mu, sigma      = np.mean(buf), np.std(buf)
            regime_warning = abs(delta_alpha) > mu + self.regime_sigma * max(sigma, 1e-4)
        else:
            regime_warning = False

        stop = self.stop_loss_multiplier * self.entry_z

        # Entry logic
        if self.position == 0:
            self.bars_in_trade = 0 # Reset counter
            # Block new entries during regime shifts if filter is on
            if self.regime_filter and regime_warning:
                pass   # Stay flat
            elif z > self.entry_z: self.position = -1 # Spread too high -> short
            elif z < -self.entry_z: self.position = +1 # spread too low -> long
        # Exit logic
        else:
            self.bars_in_trade += 1
            # Target (hysteresis) — only allowed after minimum holding period
            min_hold_bars = max(1, int(self.min_hold_frac * half_life))
            held_long_enough = self.bars_in_trade >= min_hold_bars
            reached_target = held_long_enough and (
                (self.position == +1 and z > -self.exit_z) or
                (self.position == -1 and z < self.exit_z)
            )
            # Time decay — always allowed (prevents stale positions)
            time_decay = self.bars_in_trade > (2.0 * half_life)
            # Stop-loss — always allowed (risk management must never be blocked)
            hit_stop = (self.position == +1 and z < -stop) or \
                       (self.position == -1 and z > stop)
            if reached_target or time_decay or hit_stop:
                self.position = 0
        return self.position, regime_warning

# 6. Define portfolio
class Portfolio:
    """
    Volatility-targeted dollar-neutral portfolio.

    At each signal entry:
      - Allocates `capital` across legs using hedge-ratio weights
      - Scales position size so expected daily spread vol = target_vol
      - Models transaction costs as a fixed fraction of notional traded
        (not of PnL), applied at entry and exit

    Attributes
    ----------
    capital    : total capital in currency units (e.g. USD)
    target_vol : target daily portfolio volatility as a fraction (e.g. 0.01 = 1%)
    cost_bps   : one-way transaction cost in basis points (e.g. 5 = 0.05%)
    """
    def __init__(self, cfg: Config = None,
                 capital: float = None,
                 target_vol: float = None,
                 cost_bps: float = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to Portfolio.")
        self.cfg = cfg
        self.capital   = capital   or cfg.capital
        self.target_vol = target_vol or cfg.target_vol
        self.cost_bps  = cost_bps  or cfg.cost_bps
        self.records   = []
        self.position  = 0
        self.last_cost = 0.0
        self.last_weights = None


    def compute_weights(self, beta: np.ndarray,
                        spread_vol: float) -> np.ndarray:
        """
        Returns weight vector w = [w_dep, w_indep_1, ..., w_indep_n].

        Step 1 — hedge ratio: long $1 of dependent asset, short $beta of independent assets
        Step 2 — normalise: sum(|w|) = 1  → weights are fractions of capital
        Step 3 — vol scale: multiply by target_vol / spread_vol so that a
                 1-sigma spread move produces target_vol return on capital
        """
        w     = np.empty(len(beta) + 1)
        w[0]  = 1.0
        w[1:] = -beta
        w    /= np.abs(w).sum()                          # step 2
        if spread_vol > 1e-10:
            w *= self.target_vol / spread_vol            # step 3
        return w

    def compute_shares(self, weights: np.ndarray,
                       prices: np.ndarray) -> np.ndarray:
        """
        Converts weight vector to number of shares per leg.

        shares_i = capital * weight_i / price_i

        Positive = long, negative = short.
        """
        notional = self.capital * weights                # dollars per leg
        return notional / prices                         # shares per leg

    def notional_traded(self, weights: np.ndarray) -> float:
        """Total gross notional = sum of absolute dollar values per leg."""
        return float(np.abs(weights).sum() * self.capital)

    def transaction_cost(self, weights: np.ndarray) -> float:
        """
        One-way cost = cost_bps/10000 * gross notional.
        Applied once at entry and once at exit.
        """
        return (self.cost_bps / 10_000) * self.notional_traded(weights)

    def update(self, log_returns: np.ndarray,
               weights: np.ndarray,
               signal: int,
               prices: np.ndarray,
               prev_signal: int) -> dict:
        """
        Computes PnL and costs for one bar.

        Parameters
        ----------
        log_returns : log-return of each asset this bar
        weights     : from compute_weights()
        signal      : current position (-1, 0, +1)
        prices      : current raw prices for share calculation
        prev_signal : signal from the previous bar (to detect transitions)

        Returns a dict with all metrics for this bar.
        """
        # P&L in return units and dollars
        pnl_return = float(signal * np.dot(weights, log_returns))
        pnl_dollar = pnl_return * self.capital

        if signal == 0:
            assert pnl_dollar == 0.0, f"pnl_dollar not 0"

        # Transaction costs: charged on entry and exit transitions
        cost = 0.0
        entered = (prev_signal == 0 and signal != 0)
        exited  = (prev_signal != 0 and signal == 0)
        # We use 'weights' from the current bar for entry, 
        # but for exit, we must use weights from the PREVIOUS bar (what we actually hold).
        if entered:
            cost = self.transaction_cost(weights)
            self.last_weights = weights.copy()
        elif exited:
            cost = self.transaction_cost(self.last_weights) if self.last_weights is not None else 0.0
        
        # Share breakdown per leg
        is_active = (signal != 0)
        shares = self.compute_shares(weights * signal, prices)

        record = {
            self.cfg.PNL_RETURN: (pnl_dollar - cost) / self.capital,
            self.cfg.PNL_DOLLAR: pnl_dollar - cost,
            self.cfg.PNL_GROSS:  pnl_dollar,
            self.cfg.COST: cost,
            self.cfg.NOTIONAL: self.notional_traded(weights) if is_active else 0.0,
            self.cfg.SHARES_DEP: shares[0], # Dependent shares (+ = long)
            self.cfg.SHARES_INDEP: shares[1:], # Independent shares (- = short)
            self.cfg.WEIGHTS_DEP:   weights[0] * signal,
            self.cfg.WEIGHTS_INDEP: weights[1:] * signal,
            self.cfg.ENTRY_PRICE: prices[0] if entered else np.nan,
            self.cfg.EXIT_PRICE: prices[0] if exited  else np.nan,
        }

        self.records.append(record)
        self.last_cost = cost
        return record

# 7. Analyse quality of signal
class SignalQualityModel:
    """
    Random-forest classifier trained on Backtester output to gate
    low-quality signal entries.
 
    Full loop:
        Backtester.run()          -> results  (features + PnL)
        Backtester.build_labels() -> labels   (1=profitable, 0=not)
        SignalQualityModel.train()            -> fitted RF
        (live) gate_signal()                  -> filtered position
    """
    def __init__(self, cfg: Config = None,
                 n_estimators: int = None,
                 n_splits: int = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to SignalQualityModel.")
        self.cfg = cfg
        self.model = RandomForestClassifier(
            n_estimators  = n_estimators or cfg.rf_n_estimators,
            max_features  = "sqrt",
            min_samples_leaf = 10,
            class_weight  = "balanced",
            random_state  = 42,
        )
        self.n_splits     = n_splits or cfg.rf_n_splits
        self.threshold    = cfg.rf_threshold
        self.is_fitted    = False
        self.feature_cols = None
        self.feat_template = None


    def train(self, features: pd.DataFrame, labels: pd.Series):
        """
        Trains using time-series cross-validation so future data never
        leaks into training folds. Only active-signal rows are used.
        """
        self.feature_cols = list(features.columns)
        # Store a NaN template for production use
        self.feat_template = pd.Series(np.nan, index=self.feature_cols)
        mask = labels.notna()
        X    = features.loc[mask].fillna(0).values
        y    = labels.loc[mask].values.astype(int)
 
        if len(np.unique(y)) < 2:
            logging.warning(f"Warning: only one class in labels. RF not trained.")
            return
 
        logging.info(f"\nClass distribution: {np.bincount(y)}")
        logging.info(f"Positive rate: {y.mean():.3f}")

        n_splits = 2 if y.sum() < self.n_splits * 2 else self.n_splits
        logging.warning(f"Warning: only {y.sum()} positive samples for "
            f"{self.n_splits}-fold CV. Reducing to 2 folds.")
        tscv = TimeSeriesSplit(n_splits=n_splits)

        logging.info(f"Walk-forward CV:")
        for fold, (tr, va) in enumerate(tscv.split(X)):
            self.model.fit(X[tr], y[tr])
            preds  = self.model.predict(X[va])
            report = classification_report(
                y[va], preds, output_dict=True, zero_division=0)
            # Use .get() with a fallback so missing classes don't crash
            p = report.get('1', {})
            logging.info(f"  Fold {fold+1}  "
                  f"precision={p.get('precision', float('nan')):.3f}  "
                  f"recall={p.get('recall', float('nan')):.3f}  "
                  f"f1={p.get('f1-score', float('nan')):.3f}  "
                  f"support={int(p.get('support', 0))}")
 
        self.model.fit(X, y)
        self.is_fitted = True
        logging.info(f"\nRF trained on {len(y)} trades \n({y.sum()} profitable / {(1-y).sum()} not).")
 
        fi = pd.Series(self.model.feature_importances_,
                   index=self.feature_cols).sort_values(ascending=False)
        logging.info(f"\nFeature importances: {fi.round(4).to_string()}")

    def check_quality(self, feature_row: np.ndarray) -> float:
        if not self.is_fitted:
            return 1.0
        return float(self.model.predict_proba(
            feature_row.reshape(1, -1))[0, 1])
    
    def gate_signal(self, signal: int, feature_row) -> int:
        """Returns a scaled signal based on RF confidence.
            0.0 = Skip, 0.5 = Half-size, 1.0 = Full-size."""
        if signal == 0 or not self.is_fitted:
            return signal
        # accept either a pd.Series or a plain array
        if isinstance(feature_row, pd.Series):
            x = feature_row.reindex(self.feature_cols).fillna(0).values
        else:
            x = np.nan_to_num(np.array(feature_row, dtype=float))
        prob = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        # Conviction logic
        if prob >= 0.70:
            return float(signal)        # High confidence: Full size
        elif prob >= self.threshold:    # e.g., 0.55
            return signal * 0.5         # Moderate confidence: Half size
        else:
            return 0.0                  # Low confidence: Skip
    
    def save(self, filepath: str = None):
        joblib.dump({"model": self.model,
                 "threshold": self.threshold,
                 "is_fitted": self.is_fitted,
                 "feature_cols": self.feature_cols,
                 "feat_template": self.feat_template},
                filepath)
 
    @classmethod
    def load(cls, cfg: Config) -> 'SignalQualityModel':
        data           = joblib.load(cfg.qm_file)
        instance       = cls(cfg=cfg)
        instance.model        = data["model"]
        instance.threshold    = data["threshold"]
        instance.is_fitted    = data["is_fitted"]
        instance.feature_cols = data["feature_cols"]
        instance.feat_template = data["feat_template"]
        return instance

# 7b. XGBoost signal quality model
class XGBoostQualityModel:
    """
    XGBoost classifier trained on the same feature set as SignalQualityModel.
    Uses isotonic-regression calibration so gate_signal() can threshold on
    true probability rather than a raw score.

    The interface is intentionally identical to SignalQualityModel so the two
    can be swapped or stacked in StrategyExecutor without touching any other
    class.
    """

    def __init__(self, cfg: Config = None, n_splits: int = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to XGBoostQualityModel.")
        self.cfg       = cfg
        self.threshold = cfg.xgb_threshold
        self.n_splits  = n_splits or cfg.rf_n_splits
        self.is_fitted = False
        self.feature_cols  = None
        self.feat_template = None

        base = XGBClassifier(
            n_estimators    = cfg.xgb_n_estimators,
            max_depth       = cfg.xgb_max_depth,
            learning_rate   = cfg.xgb_learning_rate,
            subsample       = cfg.xgb_subsample,
            colsample_bytree= 0.8,
            use_label_encoder=False,
            eval_metric     = "logloss",
            random_state    = 42,
            verbosity       = 0,
        )
        self._base = base

    def train(self, features: pd.DataFrame, labels: pd.Series):
        self.feature_cols  = list(features.columns)
        self.feat_template = pd.Series(np.nan, index=self.feature_cols)

        mask = labels.notna()
        X    = features.loc[mask].fillna(0).values
        y    = labels.loc[mask].values.astype(int)

        if len(np.unique(y)) < 2:
            logging.warning("XGB: only one class in labels. Not trained.")
            return

        logging.info(f"\nXGB class distribution: {np.bincount(y)}")
        logging.info(f"XGB positive rate: {y.mean():.3f}")

        n_splits = max(2, min(self.n_splits, int(np.bincount(y).min())))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        logging.info("XGB Walk-forward CV:")
        for fold, (tr, va) in enumerate(tscv.split(X)):
            fold_counts = np.bincount(y[tr])
            if len(fold_counts) < 2 or fold_counts.min() < 2:
                logging.warning(f"XGB Fold {fold+1}: insufficient class support in training split, skipping.")
                continue
            calib_cv = int(min(3, fold_counts.min()))  # cast to Python int
            self.model = CalibratedClassifierCV(self._base, cv=calib_cv, method="isotonic")
            self.model.fit(X[tr], y[tr])
            preds  = self.model.predict(X[va])
            report = classification_report(y[va], preds, output_dict=True, zero_division=0)
            p = report.get('1', {})
            logging.info(
                f"  Fold {fold+1}  "
                f"precision={p.get('precision', float('nan')):.3f}  "
                f"recall={p.get('recall', float('nan')):.3f}  "
                f"f1={p.get('f1-score', float('nan')):.3f}  "
                f"support={int(p.get('support', 0))}"
            )

        # Final fit on full data — guard against insufficient minority class
        final_counts = np.bincount(y)
        if final_counts.min() < 2:
            logging.warning(
                f"XGB: minority class has only {final_counts.min()} sample(s). "
                f"Fitting without calibration."
            )
            self.model = self._base
        else:
            calib_cv = int(min(3, final_counts.min()))  # cast to Python int
            self.model = CalibratedClassifierCV(self._base, cv=calib_cv, method="isotonic")
        
        self.model.fit(X, y)
        self.is_fitted = True
        logging.info(f"XGB trained on {len(y)} trades ({y.sum()} profitable).")

        # Feature importances — handle both calibrated and uncalibrated
        try:
            base_xgb = self.model.calibrated_classifiers_[0].estimator
        except AttributeError:
            base_xgb = self.model  # uncalibrated fallback
        fi = pd.Series(
            base_xgb.feature_importances_,
            index=self.feature_cols
        ).sort_values(ascending=False)
        logging.info(f"XGB Feature importances:\n{fi.round(4).to_string()}")

    def gate_signal(self, signal: int, feature_row) -> float:
        """Identical conviction logic to SignalQualityModel.gate_signal."""
        if signal == 0 or not self.is_fitted:
            return signal
        if isinstance(feature_row, pd.Series):
            x = feature_row.reindex(self.feature_cols).fillna(0).values
        else:
            x = np.nan_to_num(np.array(feature_row, dtype=float))
        prob = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        if prob >= 0.70:
            return float(signal)
        elif prob >= self.threshold:
            return signal * 0.5
        else:
            return 0.0

    def save(self, filepath: str):
        joblib.dump({
            "model":        self.model,
            "threshold":    self.threshold,
            "is_fitted":    self.is_fitted,
            "feature_cols": self.feature_cols,
            "feat_template":self.feat_template,
        }, filepath)

    @classmethod
    def load(cls, cfg: Config) -> 'XGBoostQualityModel':
        data           = joblib.load(cfg.xgb_file)
        instance       = cls(cfg=cfg)
        instance.model         = data["model"]
        instance.threshold     = data["threshold"]
        instance.is_fitted     = data["is_fitted"]
        instance.feature_cols  = data["feature_cols"]
        instance.feat_template = data["feat_template"]
        return instance

# 7c. Koopman regime filter (EDMD-based)
class KoopmanRegimeFilter:
    """
    Extended Dynamic Mode Decomposition (EDMD) regime gate.

    Fits a linear Koopman operator on a rolling window of the spread using
    a delay-embedding observable dictionary.  The leading eigenvalue's
    modulus encodes the current mean-reversion rate:

      |lambda_1| << 1  →  fast reversion  → green light to trade
      |lambda_1| ≈ 1   →  spread near unit root  → block entry

    Architecture role: upstream gate, called BEFORE the Kalman filter
    drives signal generation.  If is_stationary() returns False, the
    calling code should skip signal generation entirely for that bar.

    Parameters (in Config)
    ----------------------
    koopman_window     : int   number of observations used per EDMD fit
    koopman_n_obs      : int   number of delay embeddings (dictionary size)
    koopman_eig_thresh : float leading |eigenvalue| ceiling (default 0.97)
    """

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.window     = cfg.koopman_window
        self.n_obs      = cfg.koopman_n_obs
        self.thresh     = cfg.koopman_eig_thresh
        self._lead_eig  = np.nan          # last estimated leading |eigenvalue|
        self._eig_hist  = []              # rolling history for diagnostics

    # ------------------------------------------------------------------
    # Core EDMD
    # ------------------------------------------------------------------
    def _build_psi(self, x: np.ndarray) -> np.ndarray:
        """
        Build the observable matrix Psi using a delay-embedding dictionary.

        For a 1-D signal x of length T, we create a (T - n_obs) × n_obs
        matrix where row t contains [x_t, x_{t-1}, ..., x_{t-n_obs+1}].
        This is the Hankel / time-delay embedding, a standard Koopman dict
        for scalar time series.
        """
        T = len(x)
        if T <= self.n_obs:
            return None
        rows = T - self.n_obs
        Psi  = np.column_stack([x[i : i + rows] for i in range(self.n_obs - 1, -1, -1)])
        return Psi   # shape (rows, n_obs)

    def _fit_koopman(self, spread_window: np.ndarray) -> float:
        """
        Solve the EDMD least-squares problem:  Psi_prime ≈ K @ Psi.T

        Returns the modulus of the leading eigenvalue of K, or nan on failure.
        """
        Psi = self._build_psi(spread_window)
        if Psi is None or Psi.shape[0] < self.n_obs + 2:
            return np.nan

        Psi_x  = Psi[:-1]   # "current"  observables  shape (M, n_obs)
        Psi_y  = Psi[1:]    # "next-step" observables  shape (M, n_obs)

        # K = argmin ||Psi_y - Psi_x @ K||_F   (transpose convention)
        # Solved via pseudo-inverse: K = pinv(Psi_x) @ Psi_y
        try:
            K      = np.linalg.lstsq(Psi_x, Psi_y, rcond=None)[0]  # (n_obs, n_obs)
            eigvals = np.linalg.eigvals(K)
            lead    = float(np.max(np.abs(eigvals)))
            return lead
        except np.linalg.LinAlgError:
            return np.nan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, spread_buffer: np.ndarray) -> float:
        """
        Update the Koopman estimate from the latest spread buffer.

        Parameters
        ----------
        spread_buffer : array of recent raw spread values (innovations e_t),
                        length >= koopman_window.

        Returns
        -------
        Leading eigenvalue modulus (float), or nan if estimation failed.
        """
        if len(spread_buffer) < self.window:
            self._lead_eig = np.nan
            return np.nan

        window_data    = np.array(spread_buffer[-self.window:], dtype=float)
        # Standardise so EDMD isn't scale-sensitive
        std = window_data.std()
        if std < 1e-10:
            self._lead_eig = np.nan
            return np.nan
        window_norm    = (window_data - window_data.mean()) / std

        self._lead_eig = self._fit_koopman(window_norm)
        self._eig_hist.append(self._lead_eig)
        return self._lead_eig

    def is_stationary(self) -> bool:
        """
        Returns True if the spread is in a mean-reverting regime.

        A NaN eigenvalue (too few data) defaults to True so the filter
        doesn't block entries during warm-up.
        """
        if np.isnan(self._lead_eig):
            return True   # warm-up: don't block
        return self._lead_eig < self.thresh

    @property
    def lead_eigenvalue(self) -> float:
        return self._lead_eig

    def diagnostics(self) -> dict:
        hist = np.array([e for e in self._eig_hist if not np.isnan(e)])
        return {
            "koopman_lead_eig_mean": float(np.mean(hist)) if len(hist) else np.nan,
            "koopman_lead_eig_std":  float(np.std(hist))  if len(hist) else np.nan,
            "koopman_block_rate":    float((hist >= self.thresh).mean()) if len(hist) else np.nan,
        }

# 7d. Walk-forward validator
class WalkForwardValidator:
    """
    Expanding-window walk-forward validation.

    Splits the training set into `n_splits` folds where the training
    window grows with each fold (no data leakage).  For each fold it:
      1. Re-fits CointegrationModel on the current training slice.
      2. Re-runs KalmanMLE to get fresh noise hyperparameters.
      3. Runs Backtester and collects performance metrics.
      4. Trains RF + XGBoost classifiers on that fold's labels.

    At the end it aggregates fold metrics and returns a summary DataFrame,
    giving you an honest estimate of out-of-sample generalisation before
    you touch the held-out test set.

    Parameters (in Config)
    ----------------------
    wf_n_splits       : int   number of folds (default 5)
    wf_min_train_frac : float minimum fraction of data used in fold 1 (default 0.40)
    """

    def __init__(self, data: DataHandler, cfg: Config,
                 beta_init: np.ndarray, alpha_init: float,
                 opt_W_beta: float, opt_W_alpha: float, opt_V: float,
                 opt_nu: float = None):
        self.data       = data
        self.cfg        = cfg
        self.beta_init  = beta_init
        self.alpha_init = alpha_init
        self.opt_W_beta  = opt_W_beta
        self.opt_W_alpha = opt_W_alpha
        self.opt_V       = opt_V
        self.opt_nu      = opt_nu
        self.fold_metrics: list[dict] = []

    def run(self) -> pd.DataFrame:
        """
        Execute all folds.  Returns a DataFrame with one row per fold
        showing Sharpe, MaxDrawdown, WinRate, NumTrades, and classifier
        CV f1 scores.
        """
        n     = len(self.data.df)
        cfg   = self.cfg
        perf  = Performance(cfg=cfg)

        min_train = max(int(n * cfg.wf_min_train_frac), cfg.bt_window)
        # Expanding fold boundaries: train ends grow from min_train to n
        fold_ends = np.linspace(min_train, n, cfg.wf_n_splits + 1, dtype=int)[1:]

        logging.info(f"\n{'='*60}")
        logging.info(f"WALK-FORWARD VALIDATION  ({cfg.wf_n_splits} folds)")
        logging.info(f"{'='*60}")

        for fold_idx, train_end in enumerate(fold_ends):
            logging.info(f"\n--- Fold {fold_idx + 1}/{cfg.wf_n_splits} "
                         f"| train rows: {train_end} ---")

            # Slice training data for this fold
            fold_df   = self.data.df.iloc[:train_end]
            fold_data = DataHandler.__new__(DataHandler)
            fold_data.log = self.data.log
            fold_data.df  = fold_df

            # Re-fit cointegration on this fold's window
            coint  = CointegrationModel(cfg=cfg)
            try:
                res    = coint.fit(fold_df.tail(cfg.bt_window))
                beta   = res["VECM_beta"][1:]
                alpha  = res["VECM_alpha"]
            except Exception as exc:
                logging.debug(f"  Fold {fold_idx+1}: coint failed ({exc}), using prior beta/alpha.")
                beta, alpha = self.beta_init, self.alpha_init

            # Re-run MLE on this fold (optional but gives fold-specific params)
            y_fold = fold_df[cfg.dep_col].values
            X_fold = add_constant(fold_df[cfg.indep_cols], prepend=False).values
            try:
                mle    = KalmanMLE(init_beta=beta, init_alpha=alpha, cfg=cfg)
                opt    = mle.fit(y_fold, X_fold)
                w_beta, w_alpha, v = opt[0], opt[cfg.state_dim - 1], opt[cfg.state_dim]
                nu     = opt[cfg.state_dim + 1] if cfg.dist_shape == 'student' else None
            except Exception as exc:
                logging.warning(f"  Fold {fold_idx+1}: MLE failed ({exc}), using global params.")
                w_beta, w_alpha, v, nu = self.opt_W_beta, self.opt_W_alpha, self.opt_V, self.opt_nu

            # Build fresh Kalman filter for this fold
            kf_fold = KalmanModel(
                W_diag=[w_beta] * (cfg.state_dim - 1) + [w_alpha],
                V=v, cfg=cfg
            )
            kf_fold.nu = nu

            # Derive entry_z from a short Kalman run on the fold data
            kf_probe = KalmanModel(
                W_diag=[w_beta] * (cfg.state_dim - 1) + [w_alpha],
                V=v, cfg=cfg
            )
            kf_probe.initialise(beta, alpha)
            df_hist = kf_probe.run_and_report(
                y_fold, X_fold,
                burn_in=cfg.kalman_burn_in, nu=nu,
                dates=fold_df.index
            )
            z_train = df_hist["z_score"].dropna()
            entry_z = np.percentile(np.abs(z_train), cfg.entry_z_percentile)

            # Backtest on this fold
            bt = Backtester(data=fold_data, kf=kf_fold, cfg=cfg, entry_z=entry_z)
            bt_res = None
            try:
                bt_res    = bt.run()
                fold_perf = perf.evaluate(bt_res, label=f"wf_fold_{fold_idx+1}")
            except Exception as exc:
                logging.warning(
                    f"  Fold {fold_idx+1}: backtest or evaluation failed ({exc}). "
                    f"Recording NaN sentinel row."
                )
                fold_perf = {
                    "Sharpe": np.nan, "Calmar": np.nan, "MaxDrawdown": np.nan,
                    "TotalReturn": np.nan, "AnnReturn": np.nan, "Turnover": np.nan,
                    "WinRate": np.nan, "NumTrades": 0,
                }

            rf_f1, xgb_f1 = np.nan, np.nan
            if bt_res is not None:
                # Train RF and XGB classifiers on this fold's labels
                labels   = bt.build_labels()
                print(f"Label value counts: {labels.value_counts(dropna=False).to_dict()}")
                features = bt.get_feature_matrix()
                print("\nFeature matrix stats:")
                print(features.describe().round(6))
                print("\nNaN counts:", features.isna().sum().to_dict())
                print("Constant cols:", [c for c in features.columns if features[c].nunique() <= 1])

                valid_labels = labels.dropna()
                if (len(valid_labels) >= 6 
                        and len(np.unique(valid_labels)) == 2
                        and np.bincount(valid_labels.astype(int).values).min() >= 2):
                    # RF
                    rf = SignalQualityModel(cfg=cfg)
                    rf.train(features, labels)
                    common_idx = features.index.intersection(labels.dropna().index)
                    X_full = features.loc[common_idx].fillna(0).values
                    y_full = labels.loc[common_idx].fillna(0).astype(int).values
                    rf_preds  = rf.model.predict(X_full)
                    rf_rep    = classification_report(y_full, rf_preds, output_dict=True, zero_division=0)
                    rf_f1 = rf_rep.get('1', {}).get('f1-score', np.nan)

                    # XGB
                    xgb = XGBoostQualityModel(cfg=cfg)
                    xgb.train(features, labels)
                    try:
                        xgb_proba = xgb.model.predict_proba(X_full)[:, 1]
                        xgb_preds = (xgb_proba >= 0.5).astype(int)
                    except AttributeError:
                        xgb_preds = (xgb.model.predict(X_full) > 0.5).astype(int)
                    xgb_rep   = classification_report(
                        y_full, xgb_preds,
                        output_dict=True, zero_division=0
                    )
                    xgb_f1 = xgb_rep.get('1', {}).get('f1-score', np.nan)

            self.fold_metrics.append({
                "fold":        fold_idx + 1,
                "train_rows":  train_end,
                "entry_z":     entry_z,
                **fold_perf,
                "rf_f1":       rf_f1,
                "xgb_f1":      xgb_f1,
            })

        summary = pd.DataFrame(self.fold_metrics)
        self._print_summary(summary)
        self._plot_wf_equity(summary)
        return summary

    def _print_summary(self, summary: pd.DataFrame):
        logging.info(f"\n{'='*60}")
        logging.info("WALK-FORWARD SUMMARY")
        logging.info(f"{'='*60}")
        cols = ["fold", "Sharpe", "MaxDrawdown", "WinRate", "NumTrades", "rf_f1", "xgb_f1"]
        available = [c for c in cols if c in summary.columns]
        logging.info(f"\n{summary[available].round(4).to_string(index=False)}")
        logging.info(f"\nMean Sharpe:  {summary['Sharpe'].mean():.4f}  "
                     f"± {summary['Sharpe'].std():.4f}")
        logging.info(f"Mean MaxDD:   {summary['MaxDrawdown'].mean():.4f}")
        if 'rf_f1' in summary:
            logging.info(f"Mean RF F1:   {summary['rf_f1'].mean():.4f}")
        if 'xgb_f1' in summary:
            logging.info(f"Mean XGB F1:  {summary['xgb_f1'].mean():.4f}")

    def _plot_wf_equity(self, summary: pd.DataFrame):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].bar(summary["fold"], summary["Sharpe"], color="steelblue")
        axes[0].axhline(summary["Sharpe"].mean(), color="red", linestyle="--",
                        label=f"Mean={summary['Sharpe'].mean():.2f}")
        axes[0].set_title("Walk-Forward Sharpe by Fold")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "rf_f1" in summary.columns and "xgb_f1" in summary.columns:
            axes[1].plot(summary["fold"], summary["rf_f1"],  "o-", label="RF F1")
            axes[1].plot(summary["fold"], summary["xgb_f1"], "s-", label="XGB F1")
            axes[1].set_title("Classifier F1 (class=1 / profitable trades) by Fold")
            axes[1].set_ylabel("F1 Score")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fpath = os.path.join(self.cfg.output_dir, "walk_forward_summary.png")
        plt.savefig(fpath, dpi=150)
        plt.close()
        logging.info(f"Walk-forward plot saved to {fpath}")

# 8. Execute strategy
class StrategyExecutor:
    """
    Engine to execute one step of the trading strategy.

    Gate order (each gate can block entry independently):
      0. Koopman regime gate  — upstream stationarity check
      1. SignalEngine         — z-score entry / exit logic
      2. RF quality gate      — Random Forest conviction filter
      3. XGB quality gate     — XGBoost conviction filter (optional second gate)
    """
    def __init__(self, kf, signal_engine, portfolio,
                 quality_model=None,
                 xgb_model=None,
                 koopman_filter=None,
                 cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to StrategyExecutor.")
        self.cfg            = cfg
        self.kf             = kf
        self.signal_engine  = signal_engine
        self.portfolio      = portfolio
        self.quality_model  = quality_model   # RF
        self.xgb_model      = xgb_model       # XGBoost (optional)
        self.koopman_filter = koopman_filter   # KoopmanRegimeFilter (optional)

    def execute_step(self, y_t, X_t, log_ret, raw_prices, prev_signal, external_features: dict = {}):
        """
        Processes a single bar of data.
        
        Parameters:
            y_t, X_t: Model inputs for the Kalman Filter.
            log_ret: Asset returns for PnL calculation.
            raw_prices: Actual prices for share sizing.
            prev_signal: The position held in the previous bar.
            external_features: Optional dict (e.g., Johansen trace) to merge into RF features.
        """
        # 0. Koopman upstream regime gate
        #    Uses the Kalman filter's accumulated innovation history as the
        #    spread buffer — no additional computation needed.
        koopman_blocked = False
        koopman_eig     = np.nan
        if self.koopman_filter is not None:
            koopman_eig = self.koopman_filter.update(self.kf._innov_history)
            koopman_blocked = not self.koopman_filter.is_stationary()

        # 1. Update Kalman Filter State
        prev_m = self.kf.m.copy()
        sr = self.kf.step(y_t, X_t)
        
        # 2. Extract Internal Features
        delta_alpha = sr.m[-1] - prev_m[-1]
        beta_vel = np.linalg.norm(sr.m[:-1] - prev_m[:-1])
        spread_vol = float(np.sqrt(sr.Q)) * sr.z_std # That's because z_score = e/(sqrt(Q)*sigma_z), and e is the co-integration spread
        
        features = {
            self.cfg.Z_SCORE: sr.z_score,
            self.cfg.LOG_Q: np.log(sr.Q),
            self.cfg.DELTA_ALPHA: delta_alpha,
            self.cfg.BETA_VELOCITY: beta_vel,
            self.cfg.SPREAD_VOL: spread_vol,
            self.cfg.HALF_LIFE: external_features.get(self.cfg.HALF_LIFE, 0),
            self.cfg.Z_VELOCITY: sr.z_score - (self.kf._z_score_history[-2] if len(self.kf._z_score_history)>1 else sr.z_score),
            self.cfg.KOOPMAN_EIG: koopman_eig,
        }
        if external_features:
            features.update(external_features)

        # 3. Generate Raw Signal
        signal, regime_warn = self.signal_engine.generate(z=sr.z_score,
                                                          delta_alpha=delta_alpha,
                                                          half_life=features.get(self.cfg.HALF_LIFE, 20.0))

        # 3b. Block new entries when Koopman says spread is non-stationary.
        #     We only block entries (signal != 0 appearing from flat); open
        #     positions are allowed to exit normally so we don't trap risk.
        if koopman_blocked and prev_signal == 0 and signal != 0:
            signal = 0
        
        # 4. Apply Machine Learning Gate (RF)
        if self.quality_model and self.quality_model.is_fitted and signal != 0:
            signal = self.quality_model.gate_signal(signal, pd.Series(features))

        # 4b. Apply XGBoost quality gate (second independent filter)
        #     Both gates must independently approve the trade.  The minimum
        #     conviction of the two is used so neither can be overridden.
        if self.xgb_model and self.xgb_model.is_fitted and signal != 0:
            xgb_signal = self.xgb_model.gate_signal(signal, pd.Series(features))
            # Take the more conservative of the two conviction scalings
            if signal != 0:
                rf_scale  = abs(signal) / abs(signal) if signal != 0 else 0
                xgb_scale = abs(xgb_signal) / abs(signal) if signal != 0 else 0
                min_scale = min(rf_scale, xgb_scale)
                signal    = np.sign(signal) * abs(signal) * min_scale
            
        # 5. Portfolio Allocation & PnL Update
        weights = self.portfolio.compute_weights(sr.m[:-1], spread_vol=spread_vol)
        port_record = self.portfolio.update(
            log_ret, weights, signal, prices=raw_prices, prev_signal=prev_signal
        )
        
        # 6. Consolidate for Recording
        return {
            **port_record,
            **features,
            self.cfg.REGIME_WARNING: int(regime_warn),
            self.cfg.KOOPMAN_BLOCKED:  int(koopman_blocked),
            self.cfg.SIGNAL: signal,
        }
      
# 9. Backtest strategy
class Backtester:
    """
    Walk-forward backtest that generates the labelled dataset needed to
    train SignalQualityModel.
 
    Design decisions
    ----------------
    - The Kalman filter is initialised once from the first VECM fit and
      runs continuously. The rolling VECM supplies features only; it does
      NOT reset the filter (which would discard accumulated state).
    - Labels are built retrospectively: entry at time t is labelled 1 if
      cumulative PnL over the next `label_horizon` steps is positive.
    - Feature columns stay in sync with SignalQualityModel.FEATURE_COLS.
    """

    def __init__(self, data: DataHandler, kf: KalmanModel,
                 entry_z: float, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to Backtester.")
        self.cfg = cfg
        self.data          = data
        self.kf            = kf
        self.window        = cfg.bt_window
        self.label_horizon = cfg.label_horizon
        self.DEP_COL       = cfg.dep_col
        self.INDEP_COLS    = cfg.indep_cols
        self.signal_engine = SignalEngine(entry_z=entry_z, cfg=cfg)
        self.portfolio     = Portfolio(cfg=cfg)
        self.test_engine   = StatisticalTestEngine()
        self.coint         = CointegrationModel(cfg=cfg)
        self.results       = None

    def run(self, quality_model=None, xgb_model=None,
            koopman_filter=None) -> pd.DataFrame:
        """
        Run the backtest.

        Parameters
        ----------
        quality_model  : fitted SignalQualityModel (RF), or None
        xgb_model      : fitted XGBoostQualityModel, or None
        koopman_filter : KoopmanRegimeFilter instance, or None
        """
        df = self.data.df
        n = len(df)
        if n <= self.window:
            raise ValueError(
                f"Fold data ({n} rows) is shorter than bt_window "
                f"({self.window}). Increase wf_min_train_frac or reduce bt_window."
            )
        
        executor = StrategyExecutor(
            kf=self.kf,
            signal_engine=self.signal_engine,
            portfolio=self.portfolio,
            quality_model=quality_model,
            xgb_model=xgb_model,
            koopman_filter=koopman_filter,
            cfg=self.cfg,
        )
        kalman_init = False
        records = []
        skipped = 0
        po_stats = []

        logging.info(f"First backtest bar: {df.index[self.window]}")
        logging.info(f"Last backtest bar:  {df.index[-1]}")

        for t in range(self.window, n):
            df_window = self.data.get_window(t, self.window)

            # Cointegration fit for features (does NOT reset the filter)
            try:
                coint_res      = self.coint.fit(df_window)
                po_stats.append(coint_res["PO_p_value"])
                beta_vecm      = coint_res["VECM_beta"][1:]
                alpha_vecm     = coint_res["VECM_alpha"]
                ext_feat = {
                    self.cfg.JOHANSEN_TRACE: coint_res["johansen"].trace_stat[0],
                    self.cfg.HALF_LIFE: coint_res["half_life"]
                }
                coint_ok = True
            except Exception as exc:
                logging.debug(f"  Coint failed at t={t}: {exc}")
                skipped += 1
                coint_ok = False
                ext_feat = {
                    self.cfg.JOHANSEN_TRACE: 0.0,
                    self.cfg.HALF_LIFE:      0.0,
                }
                continue

            # Initialise filter once from first window
            if not kalman_init:
                if not coint_ok:
                    continue  # can't init without valid coint
                self.kf.initialise(beta_vecm, alpha_vecm)
                kalman_init = True
                self.kf._innov_history   = []   # start clean
                self.kf._z_score_history = []
                continue

            # Always step the Kalman filter, even on coint failure
            y_t, X_t = self.data.get_observation(
                self.cfg.dep_col, self.cfg.indep_cols, idx=t)

            # PnL
            if t + 1 >= n:
                continue   # no next bar to trade into
            log_ret    = df.iloc[t + 1].values - df.iloc[t].values      # next bar return
            raw_prices = np.exp(df.iloc[t + 1][[self.cfg.dep_col]           # next bar price
                                + self.cfg.indep_cols].values)
            #log_ret = df.iloc[t].values - df.iloc[t - 1].values
            prev_signal = records[-1][self.cfg.SIGNAL] if records else 0
            # raw prices needed for share calculation (not log-prices)
            #raw_prices = np.exp(df.iloc[t][[self.cfg.dep_col] + self.cfg.indep_cols].values)
            step_record = executor.execute_step(y_t, X_t, log_ret, raw_prices, prev_signal, ext_feat)
            step_record[self.cfg.datetime_col] = df.index[t]

            records.append(step_record)

        logging.info(f"PO p-value distribution: mean={np.mean(po_stats):.3f} "
             f"median={np.median(po_stats):.3f} "
             f"pct<0.05={np.mean(np.array(po_stats)<0.05):.1%}")
        
        logging.info(f"Total bars skipped due to coint failure: {skipped}/{n}")
        if not records:
            raise ValueError("No valid observations generated.")
 
        self.results = pd.DataFrame(records).set_index(str(self.cfg.datetime_col))
        return self.results
    
    def build_labels(self) -> pd.Series:
        """
        Label = 1 if cumulative PnL over next `label_horizon` steps > 0.
        Flat periods (signal == 0) receive NaN and are excluded from RF.
        """
        if self.results is None: raise RuntimeError("Run backtest first.")
 
        pnl    = self.results[self.cfg.PNL_RETURN]
        signal = self.results[self.cfg.SIGNAL]
        labels = pd.Series(np.nan, index=self.results.index, name="label")

        # Exclude burn-in rows
        valid_mask = self.results[self.cfg.Z_SCORE].notna()
 
        # Identify trade entry points (where signal changes from 0)
        for i in range(len(signal) - self.label_horizon):
            if not valid_mask.iloc[i]:
                continue
            if signal.iloc[i] != 0 and (i == 0 or signal.iloc[i-1] == 0):
                # Sum net PnL over the holding horizon
                fwd_pnl = pnl.iloc[i : i + self.label_horizon].sum()
                
                # LABEL 1 only if the trade was actually profitable AFTER all costs
                labels.iloc[i] = 1 if fwd_pnl > 0 else 0

        return labels

    def get_feature_matrix(self) -> pd.DataFrame:
        """Returns the feature columns in SignalQualityModel.FEATURE_COLS order."""
        if self.results is None:
            raise RuntimeError("Call run() before get_feature_matrix().")
        feat_cols = [c for c in self.results.columns if c.startswith("feat_")]
        # Exclude burn-in rows where z_score is NaN
        mask = self.results[self.cfg.Z_SCORE].notna()
        return self.results.loc[mask, feat_cols]
 
    def apply_quality_filter(self, qm=None, xgb_model=None) -> pd.DataFrame:
        """
        Returns a copy of self.results with signal and PnL recomputed
        after applying the RF and/or XGBoost quality gate(s).

        Both gates are applied independently; the minimum conviction of
        the two is used (same logic as StrategyExecutor).
        """
        features  = self.get_feature_matrix()
        raw_pnl   = self.results[self.cfg.PNL_RETURN]
        raw_sig   = self.results[self.cfg.SIGNAL]
 
        new_signal = raw_sig.copy().astype(float)
        for idx_label in raw_sig.index:
            s = raw_sig.loc[idx_label]
            if s == 0:
                continue
            feat = features.loc[idx_label]

            # RF gate
            s_rf  = qm.gate_signal(s, feat)      if (qm  and qm.is_fitted)  else s
            # XGB gate
            s_xgb = xgb_model.gate_signal(s, feat) if (xgb_model and xgb_model.is_fitted) else s

            # Take the more conservative conviction
            scale_rf  = abs(s_rf)  / abs(s) if s != 0 else 0
            scale_xgb = abs(s_xgb) / abs(s) if s != 0 else 0
            new_signal.loc[idx_label] = np.sign(s) * abs(s) * min(scale_rf, scale_xgb)
 
        # New PnL is scaled by the conviction (e.g., 0.5x signal = 0.5x PnL)
        # We use (new_signal / raw_sig) to get the scaling factor (0, 0.5, or 1)
        scaling_factor = (new_signal / raw_sig).fillna(0)
        new_pnl = raw_pnl * scaling_factor
 
        filtered = self.results.copy()
        filtered[self.cfg.SIGNAL] = new_signal
        filtered[self.cfg.PNL_RETURN] = new_pnl
        return filtered

# 10. Analyse performance
class Performance:
    """Strategy evaluation: Sharpe, Calmar, max drawdown, turnover, win rate."""
    def __init__(self, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to Performance.")
        self.cfg = cfg

    def evaluate(self, results: pd.DataFrame, label: str = "") -> dict:
        pnl      = results[self.cfg.PNL_RETURN]
        signal   = results[self.cfg.SIGNAL]
        cum      = pnl.cumsum()
        drawdown = cum - cum.cummax()

        # Annualised return — infer years from index if DatetimeIndex
        if isinstance(results.index, pd.DatetimeIndex):
            years = (results.index[-1] - results.index[0]).days / 365.25
        else:
            years = len(results) / 252   # fallback: assume daily
        ann_return = cum.iloc[-1] / years if years > 0 else np.nan

        turnover = (signal.diff().abs() > 0).mean()
        sharpe   = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
        max_dd   = drawdown.min()
        calmar   = ann_return / abs(max_dd) if max_dd != 0 else np.nan
        active   = pnl[signal != 0]
 
        metrics = {
            "Sharpe":      sharpe,
            "Calmar":      calmar,
            "MaxDrawdown": max_dd,
            "TotalReturn": cum.iloc[-1],
            "AnnReturn":  ann_return,
            "Turnover":    turnover,
            "WinRate":     (active > 0).mean() if len(active) else np.nan,
            "NumTrades":   int((signal.diff().abs() > 0).sum()),
        }
 
        tag = f" [{label}]" if label else ""
        print(f"\nPerformance{tag}")
        for k, v in metrics.items():
            logging.info(f"  {k:<15} {v:.4f}")
 
        plt.figure(figsize=(12, 4))
        cum.plot(title=f"Cumulative PnL{tag}", grid=True)
        plt.tight_layout()
        fname = f"equity_curve{'_' + label if label else ''}.png"
        path = os.path.join(self.cfg.output_dir, fname)
        plt.savefig(path)
        plt.close()
 
        return metrics

    def compare(self, bt: 'Backtester', qm=None, xgb_model=None):
        """Side-by-side Sharpe / MaxDD before and after ML gating."""
        raw      = bt.results
        filtered = bt.apply_quality_filter(qm=qm, xgb_model=xgb_model)
        print("\n=== Before ML filter ===")
        self.evaluate(raw, label="raw")
        print("\n=== After ML filter ===")
        self.evaluate(filtered, label="filtered")
        

    def compare_df(self, *dfs: pd.DataFrame, labels: list = None):
        # Set default labels if not provided
        labels = labels or [f"df{i+1}" for i in range(len(dfs))]
        # Evaluate performance for each DataFrame
        metrics = [self.evaluate(df, label=lbl) for df, lbl in zip(dfs, labels)]
        rows = ["Sharpe", "Calmar", "MaxDrawdown", "TotalReturn",
                "AnnReturn", "WinRate", "Turnover", "NumTrades"]
        # Header
        header = f"{'Metric':<15}" + "".join(f"{lbl:>14}" for lbl in labels)
        print(f"\n{header}")
        print("-" * (15 + 14 * len(labels)))
        # Print metrics for each row
        for row in rows:
            vals = "".join(f"{m.get(row, float('nan')):>14.4f}" for m in metrics)
            logging.info(f"{row:<15}{vals}")

        # Plot and analyze each DataFrame
        for df, lbl in zip(dfs, labels):
            self.plot_z_linearity(df, label=lbl)
            self.calculate_ic_decay(df)
            self.run_shuffled_test(df, label=lbl)
            
    def plot(self, results: pd.DataFrame, label: str = ""):
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        pnl      = results[self.cfg.PNL_RETURN]
        cum      = pnl.cumsum()
        drawdown = cum - cum.cummax()
        signal   = results[self.cfg.SIGNAL]
        rolling_sharpe = (pnl.rolling(63).mean()
                        / pnl.rolling(63).std() * np.sqrt(252))

        # 1. Cumulative PnL
        axes[0].plot(cum.index, cum.values)
        axes[0].set_title(f"Cumulative PnL [{label}]")
        axes[0].set_ylabel("Cumulative log-return")
        axes[0].grid(True, alpha=0.3)

        # 2. Drawdown
        axes[1].fill_between(drawdown.index, drawdown.values, 0,
                            color="red", alpha=0.4)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        # 3. Rolling 63-bar Sharpe
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, color="green")
        axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[2].set_title("Rolling Sharpe (63 bars)")
        axes[2].set_ylabel("Sharpe")
        axes[2].grid(True, alpha=0.3)

        # 4. Signal and position
        axes[3].plot(signal.index, signal.values,
                    linewidth=0.5, color="purple")
        axes[3].set_title(self.cfg.SIGNAL)
        axes[3].set_ylabel("Position")
        axes[3].set_yticks([-1, 0, 1])
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{self.cfg.output_dir}_performance_{label}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        logging.info(f"Performance plot saved to {fname}")

    def plot_z_linearity(self, results: pd.DataFrame, label: str = "", bins: int = 5):
        """
        Check if stronger Z-scores lead to higher average returns.
        Belongs in Research Diagnostics.
        """
        df = results.copy()
        # Use absolute Z-score to measure the 'strength' of the signal
        # We assume the Z-score column is available in results via config
        z_col = self.cfg.Z_SCORE # Ensure this is in your Config class
        df['abs_z'] = df[z_col].abs()
        
        # Create quintiles
        df['z_bin'] = pd.qcut(df['abs_z'], bins, 
                             labels=[f'Q{i+1}' for i in range(bins)])
        
        linearity = df.groupby('z_bin', observed=True)[self.cfg.PNL_RETURN].mean()
        
        plt.figure(figsize=(8, 5))
        linearity.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"Z-Score Linearity [{label}]")
        plt.ylabel("Mean Bar Return")
        plt.xlabel("Z-Score Magnitude Quintile")
        plt.tight_layout()
        
        fname = os.path.join(self.cfg.output_dir,f"z_linearity_{label}.png")
        plt.savefig(fname)
        plt.close()
        logging.info(f"Z-Linearity plot saved to {fname}")
        return linearity

    def calculate_ic_decay(self, results: pd.DataFrame, horizons=[1, 2, 5, 10, 20]):
        """
        Calculate Information Coefficient (Spearman Rank Correlation) across horizons.
        Measures predictive power decay.
        """
        z_col = self.cfg.Z_SCORE
        pnl_col = self.cfg.PNL_RETURN
        ic_results = {}
        
        for h in horizons:
            # We correlate current Z-score with the sum of returns over the NEXT 'h' bars
            fwd_ret = results[pnl_col].shift(-h).rolling(window=h).sum()
            
            # Spearman is preferred as it is less sensitive to outliers
            ic = results[z_col].corr(fwd_ret, method='spearman')
            ic_results[f"H={h}"] = ic
            
        print(f"\nInformation Coefficient (IC) Decay:")
        for h, val in ic_results.items():
            logging.info(f"  {h}: {val:.4f}")
            
        return ic_results

    def run_shuffled_test(self, results: pd.DataFrame, iterations: int = 100, label: str = ""):
        """
        Monte Carlo test: scramble returns to see if PnL is due to timing or just drift.
        """
        real_pnl = results[self.cfg.PNL_RETURN].values
        real_cum = np.cumsum(real_pnl)
        
        plt.figure(figsize=(10, 6))
        
        # Run iterations
        shuffled_sharpes = []
        for _ in range(iterations):
            shuffled = np.random.permutation(real_pnl)
            shuffled_cum = np.cumsum(shuffled)
            plt.plot(shuffled_cum, color='gray', alpha=0.1)
            
            # Calculate Sharpe for this shuffle
            shuffled_sharpes.append(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252))
            
        plt.plot(real_cum, color='blue', linewidth=2, label='Actual Strategy')
        plt.title(f"Shuffled Test ({iterations} iterations) [{label}]")
        plt.ylabel("Cumulative Return")
        plt.legend()
        
        fname = os.path.join(self.cfg.output_dir,f"shuffled_test_{label}.png")
        plt.savefig(fname)
        plt.close()
        
        real_sharpe = np.mean(real_pnl) / np.std(real_pnl) * np.sqrt(252)
        percentile = (np.array(shuffled_sharpes) < real_sharpe).mean()
        
        logging.info(f"\nShuffled Test [{label}]:")
        logging.info(f"  Actual Sharpe: {real_sharpe:.4f}")
        logging.info(f"  Strategy Percentile vs Random: {percentile*100:.1f}%")

def run_production(data: DataHandler,
                   cfg: Config,
                   replay: bool = False) -> pd.DataFrame:
    """
    Steps the Kalman filter forward over all unseen dates.
    replay=True  : runs without saving state (safe to re-run anytime).
    replay=False : advances kf.last_date and persists state after each step.
    Returns a DataFrame with the same schema as Backtester.run().
    Uses the DataHandler's combined data to ensure ADF windows have sufficient history.
    """
    # ------------------------------------------------------------------
    # Initialise all components from persisted state
    # ------------------------------------------------------------------
    try:
        # Load existing state and parameters
        kf = KalmanModel.load_state(cfg=cfg)
    except Exception as e:
        logging.error(f"Could not load Kalman state: {e}")
        return pd.DataFrame()

    logging.info(f"Kalman state loaded from {cfg.state_file} "
                 f"(last_date={kf.last_date})")

    port = Portfolio(cfg=cfg)
    coint_model = CointegrationModel(cfg=cfg)
    se = SignalEngine(entry_z=kf.entry_z, cfg=cfg)
    koopman      = KoopmanRegimeFilter(cfg=cfg)

    # Load fitted ML models. gate_signal is a no-op if is_fitted=False,
    # so passing None is safe, but loading from disk is preferred.
    qm = None
    if Path(cfg.qm_file).exists():
        try:
            qm = SignalQualityModel.load(cfg=cfg)
            logging.info(f"RF model loaded (threshold={qm.threshold:.2f}, "
                         f"fitted={qm.is_fitted})")
        except Exception as e:
            logging.warning(f"Could not load RF model: {e}")

    xgb_model = None
    if Path(cfg.xgb_file).exists():
        try:
            xgb_model = XGBoostQualityModel.load(cfg=cfg)
            logging.info(f"XGB model loaded (threshold={xgb_model.threshold:.2f}, "
                         f"fitted={xgb_model.is_fitted})")
        except Exception as e:
            logging.warning(f"Could not load XGB model: {e}")

    executor = StrategyExecutor(
        kf=kf,
        signal_engine=se,
        portfolio=port,
        quality_model=qm,
        xgb_model=xgb_model,
        koopman_filter=koopman,
        cfg=cfg,
    )

    # ------------------------------------------------------------------
    # Determine which bars to process
    # ------------------------------------------------------------------

    if replay:
        # Re-run exactly the test period, start where training ended
        start_idx = data.train_test_boundary
        to_process = data.df.iloc[start_idx:]
    else:
        if kf.last_date is None:
            # If no state, start at the earliest possible window
            start_idx = cfg.bt_window
            to_process = data.df.iloc[start_idx:]
        else:
            # Start from the day after the last recorded state
            to_process = data.df[data.df.index > kf.last_date]
            if to_process.empty:
                logging.info("No new data to process.")
                return pd.DataFrame()
    
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    records = []
    z_history = list(kf._z_score_history)
    for current_date, _ in to_process.iterrows():
        # locate the integer position so get_observation can slice by idx
        idx = data.df.index.get_loc(current_date)
        df_window = data.get_window(idx, cfg.bt_window)
        if len(df_window) < 5: 
            logging.error(f"Skipping {current_date}: Window size {len(df_window)} too small for ADF.")
            continue
        y_t, X_t = data.get_observation(cfg.dep_col, cfg.indep_cols, idx=idx)

        # Health check: Kalman spread vs VECM-theoretical spread stationarity
        p_traded = coint_model.check_health(df_window, kf.m[:-1], kf.m[-1])

        # Feature extraction for ML gates
        features_ok = False
        try:
            c_res = coint_model.fit(df_window)
            p_theoretical = c_res["Spread_pvalue"]
            logging.info(f"Date: {current_date} | Traded P: {p_traded:.4f} | Theoretical P: {p_theoretical:.4f} | Drift: {abs(p_traded - p_theoretical):.4f}")
            ext_feat = {
                "feat_johansen_trace": c_res["johansen"].trace_stat[0],
                cfg.HALF_LIFE: c_res.get("half_life", 20.0)
            }
            features_ok = True
        except Exception as e:
            logging.warning(
                f"{current_date}: feature extraction failed ({e}). "
                f"ML gates will receive zero features — entries blocked by convention."
            )
            ext_feat = {
                cfg.JOHANSEN_TRACE: 0.0,
                cfg.HALF_LIFE: 0.0
            }

        # P&L uses next-bar returns
        if idx + 1 >= len(data.df):
            continue
        log_ret    = data.df.iloc[idx + 1].values - data.df.iloc[idx].values
        raw_prices = np.exp(data.df.iloc[idx + 1][[cfg.dep_col] + cfg.indep_cols].values)
        #log_ret     = data.df.iloc[idx].values - data.df.iloc[idx - 1].values
        prev_signal = records[-1][cfg.SIGNAL] if records else 0
        #raw_prices  = np.exp(data.df.iloc[idx][[dep_col] + indep_cols].values)

        # Generate signal
        step_record = executor.execute_step(y_t, X_t, log_ret, raw_prices, prev_signal, external_features=ext_feat)
        step_record[cfg.datetime_col] = current_date
        records.append(step_record)

        # Append z_t
        z_history.append(step_record[cfg.Z_SCORE])

        # Update entry for the next bar
        if len(z_history) >= cfg.kalman_norm_window:
            se.entry_z = float(np.percentile(
                np.abs(z_history[-cfg.kalman_norm_window:]),
                cfg.entry_z_percentile
            ))

        if not replay:
            kf.last_date = current_date
            kf.save_state(cfg.state_file)

        current_sig = step_record[cfg.SIGNAL]
        # Determine the conviction/size label
        conviction = abs(current_sig)
        if conviction == 1.0:
            size_label = "(FULL)"
        elif conviction > 0:
            size_label = f"(SCALED: {conviction:.1%})"
        else:
            size_label = ""

        # Determine the directional label
        if current_sig > 0:
            direction = f"LONG {cfg.dep_col} / SHORT {cfg.indep_cols[0]}"
        elif current_sig < 0:
            direction = f"SHORT {cfg.dep_col} / LONG {cfg.indep_cols[0]}"
        else:
            direction = "FLAT"

        # Print with the new dynamic action string
        if direction != "FLAT":
            logging.info(f"\n Date: {current_date} | {direction} {size_label} | Shares dep: {float(step_record[cfg.SHARES_DEP]):.2f} "
                f"| Shares indep: {step_record[cfg.SHARES_INDEP]} Notional: ${step_record[cfg.NOTIONAL]:,.2f} | features_ok={features_ok}")
    
    if not records:
        logging.info("No records generated.")
        return pd.DataFrame()
    return pd.DataFrame(records).set_index(cfg.datetime_col)

def compute_and_save_all_pairs(df_all, cfg = Config):
    """
    Compute and save co-integration analysis for all unique pairs of columns in df_all.
    """
    columns = df_all.columns.tolist()
    pairs = list(combinations(columns, 2))

    for dep_col, indep_col in pairs:
        pair_cfg = dataclasses.replace(
                cfg, dep_col=dep_col, indep_cols=[indep_col]
            )
        logging.debug(f"Pair processed: {dep_col} vs {indep_col}")
        summary_filename = os.path.join(pair_cfg.output_dir, f"summary_{dep_col}_vs_{indep_col}.csv")
        state_filename = os.path.join(pair_cfg.output_dir, f"model_state_{dep_col}_vs_{indep_col}.pkl")

        if os.path.exists(summary_filename) and os.path.exists(state_filename):
                logging.info(f"Results already exist for {dep_col} vs {indep_col}. Skipping.")
                continue

        logging.info(f"\nProcessing pair: {dep_col} vs {indep_col}")
        
        selected_columns = [dep_col, indep_col]

        # Split data
        df_raw = df_all[selected_columns]
        df_train, df_test = train_test_split(df_raw, test_size=0.3, random_state=42, shuffle=False)
        #logging.info("Data split")

        # Prepare data
        data = DataHandler(cfg=pair_cfg, df=df_train, log=pair_cfg.log_prices)
        data.summary()

        # Fit model
        model = CointegrationModel(cfg=pair_cfg)
        res = model.fit(data.df)
        logging.info("Model fit")

        confidence_metrics = model.get_model_confidence(res)
        res.update({f"CONF_{k}": v for k, v in confidence_metrics.items()})
        summary_df = pd.DataFrame.from_dict(res, orient='index', columns=['Value'])
        logging.info("Data summary created")

        # Save summary
        summary_df.to_csv(summary_filename)
        logging.info(f"Summary saved to: {summary_filename}")
    else:
        logging.info(f"Results already exist for {dep_col} vs {indep_col}. Skipping.")

def download():
    import yfinance as yf
    tickers_list=["^GSPC", "NQ=F", "EURUSD=X", "JPYUSD=X", "CHF=X", "AUDUSD=X", "NZDUSD=X"]
    prices = yf.download(tickers_list, start='2007-01-01', end='2026-04-15', group_by='ticker', auto_adjust=True, progress=False)
    if not prices.empty:
        try:
            close_prices = prices.xs('Close', axis=1, level=1)
            
            # 3. Save to CSV
            close_prices.to_csv('data.csv', index=True)
            logging.info("Success! Data saved to data_gwp.csv")
            logging.debug(close_prices.tail())
        except KeyError:
            logging.error("Error: Could not find 'Close' columns in the downloaded data.")
    else:
        logging.warning("No data was downloaded.")
    exit()

def find_first_significant_window(df_log_prices: pd.DataFrame,
                                   cfg: Config,
                                   step_size: int = 5,
                                   confirm_n: int = 2,
                                   confirm_frac: float = 0.60) -> dict | None:
    """
    Scans window lengths from coint_min_window and returns the first
    window that achieves Spread_pvalue < 0.05 AND is stable across
    confirm_n neighbouring windows on each side.

    The z_score_min_bars constraint does NOT apply here — that is
    enforced downstream in automate_strategy_params. This function
    is only responsible for finding cointegration.
    """
    total_len    = len(df_log_prices)
    # Minimum: enough rows for VECM with maxlags, not z_score_min_bars
    min_window   = max(getattr(cfg, 'coint_min_window', 60),
                       cfg.coint_maxlags * 10)
    coint_tester = CointegrationModel(cfg=cfg)

    logging.info(f"\n--- Searching for First Significant Window (p < 0.05) ---")
    logging.info(f"min_window={min_window} | total={total_len} | "
                 f"step={step_size} | confirm_n={confirm_n} | "
                 f"confirm_frac={confirm_frac}")

    def _test_window(w: int) -> float | None:
        if w < min_window or w > total_len:
            return None
        try:
            res = coint_tester.fit(df_log_prices.tail(w))
            return res.get("Spread_pvalue", 1.0)
        except Exception as e:
            logging.debug(f"Window {w} failed: {e}")
            return None

    for w in range(min_window, total_len + 1, step_size):
        p_val = _test_window(w)

        if p_val is None or p_val >= 0.05:
            if w % 100 == 0:
                logging.info(f"Window {w}: p={f'{p_val:.4f}' if p_val is not None else 'N/A'}")
            continue

        # Candidate — confirm stability across neighbours
        neighbours = [w + k * step_size
                      for k in range(-confirm_n, confirm_n + 1)
                      if k != 0]
        neighbour_results = []
        for nb in neighbours:
            nb_p = _test_window(nb)
            neighbour_results.append((nb, nb_p, nb_p is not None and nb_p < 0.05))

        n_pass   = sum(r[2] for r in neighbour_results)
        n_tested = sum(1 for r in neighbour_results if r[1] is not None)
        if n_tested < confirm_n:
            logging.info(
                f"Window {w}: only {n_tested} neighbours testable "
                f"(need {confirm_n}). Continuing search."
            )
            continue

        frac_pass = n_pass / n_tested if n_tested > 0 else 0.0

        logging.info(
            f"Candidate window {w}: p={p_val:.4f} | "
            f"neighbours passing: {n_pass}/{n_tested} ({frac_pass:.0%})"
        )

        if frac_pass >= confirm_frac:
            # Take the window with lowest p-value among all passing windows
            all_passing = [(w, p_val)] + [
                (nb, nb_p)
                for nb, nb_p, passed in neighbour_results
                if passed and nb_p is not None
            ]
            best_w, best_p = min(all_passing, key=lambda x: x[1])

            try:
                best_res = coint_tester.fit(df_log_prices.tail(best_w))
            except Exception:
                best_res = coint_tester.fit(df_log_prices.tail(w))
                best_w, best_p = w, p_val

            logging.info(
                f"Confirmed! Best window: {best_w} | p={best_p:.4f} | "
                f"half-life={best_res.get('half_life', float('nan')):.2f}"
            )
            return {
                'window_length': best_w,
                'p_value':       best_p,
                'half_life':     best_res.get('half_life'),
                'johansen_stat': best_res.get('Johansen_r=0_stat'),
                'confirm_frac':  frac_pass,
                'n_neighbours':  n_tested,
            }
        else:
            logging.info(
                f"Window {w} rejected — only {frac_pass:.0%} of neighbours "
                f"pass (need {confirm_frac:.0%}). Continuing search."
            )

    logging.error("No stable cointegration window found.")
    return None

def automate_strategy_params(cfg: Config,
                              match_results: dict) -> bool:
    """
    Sets Kalman and signal parameters from the detected cointegration
    window. Returns True on success, False if the window is too short
    to satisfy z_score constraints even after tolerance relaxation.
    """
    window    = match_results['window_length']
    half_life = match_results['half_life']

    # z_score_min_bars: recompute from current tol/conf settings
    # (may have changed since Config was constructed)
    cfg.z_score_min_bars = cfg._get_z_score_min_bar()

    # Max burn-in: 25% of window; max norm_window: 50% of window
    max_burn_in    = window // 4
    max_norm_window = window // 2

    # Relax tolerance if z_score_min_bars exceeds max_burn_in
    original_tol = cfg.z_score_std_tol
    for candidate_tol in np.arange(original_tol, 0.51, 0.01):
        cfg.z_score_std_tol  = round(float(candidate_tol), 2)
        cfg.z_score_min_bars = cfg._get_z_score_min_bar()
        if cfg.z_score_min_bars <= max_burn_in:
            break
    else:
        logging.error(
            f"window_length={window} is too short to satisfy z_score "
            f"constraints even at tol=0.50. "
            f"Minimum viable window ≈ {cfg.z_score_min_bars * 4} bars. "
            f"The search found a cointegration window but it is too short "
            f"to trade reliably. Consider using a longer price history."
        )
        return False

    if cfg.z_score_std_tol > original_tol:
        logging.warning(
            f"z_score_std_tol relaxed from {original_tol:.2f} to "
            f"{cfg.z_score_std_tol:.2f} to fit window_length={window}."
        )

    # Set parameters — all constraints now satisfiable
    cfg.kalman_norm_window = max(cfg.z_score_min_bars,
                                 min(max_norm_window, window))
    cfg.kalman_burn_in     = cfg.z_score_min_bars
    cfg.coint_maxlags      = max(5, int(np.power(window, 1/3)))
    cfg.regime_window      = max(10, int(half_life * 0.5))

    logging.info(
        f"--- Parameters Automated ---\n"
        f"  Window: {window} | Norm Window: {cfg.kalman_norm_window} | "
        f"Burn-in: {cfg.kalman_burn_in}\n"
        f"  z_score_min_bars: {cfg.z_score_min_bars} | "
        f"tol: {cfg.z_score_std_tol:.2f}\n"
        f"  Max Lags: {cfg.coint_maxlags} | "
        f"Regime Window: {cfg.regime_window}"
    )
    return True

if __name__ == "__main__":
    cfg = Config(
        price_path  = "stocks_price_history_1d_full.csv",
        dep_col      = "ISP.MI",
        indep_cols   = ["BNP.PA"],
        test_size    = 0.3,
        log_prices= True,
        dist_shape   = "gauss",
        kalman_norm_window  = 252,
        bt_window           = 252,
        label_horizon       = 5,
        stop_loss_mult      = 2.0,
        capital             = 100_000,
        cost_bps            = 5.0,
        replay = True,
        coint_enforce_i1=False,
        coint_enforce_johansen=False,
        coint_enforce_po=False
    )
    #download()

    # 1. Load data
    df_all   = pd.read_csv(cfg.price_path, index_col=cfg.datetime_col,
                           parse_dates=True).dropna()
    selected_columns = [cfg.dep_col] + cfg.indep_cols
    df_raw =  df_all[selected_columns]
    df_train_raw, df_test = train_test_split(df_raw, test_size=cfg.test_size, random_state=42, shuffle=False)

    logging.info(f"df_train_raw: {df_train_raw.index[0]} → {df_train_raw.index[-1]}")
    logging.info(f"df_test:      {df_test.index[0]} → {df_test.index[-1]}")

    # 2. Compute and save results for all pairs (if they don't exist)
    compute_and_save_all_pairs(df_all, cfg=cfg)
    exit()
    # ------------------------------------------------------------------
    # RESEARCH MODE — train filter, save state
    # ------------------------------------------------------------------

    if not os.path.exists(cfg.state_file):
        print("=" * 60)
        print("RESEARCH MODE")
        print("=" * 60)

        # Prepare data
        logging.debug("\nDataHandler for df_train_raw")
        data_raw = DataHandler(cfg=cfg, df=df_train_raw, log=cfg.log_prices)
        data_raw.summary()

        # Automatically scan all possible windows in increments of 20 days
        match = find_first_significant_window(data_raw.df, cfg, step_size=5)
        if match is None:
            logging.error("No cointegration window found.")
            exit()

        ok = automate_strategy_params(cfg, match)
        if not ok:
            logging.error("Window too short for reliable trading.")
            exit()

        cfg.bt_window = match['window_length']
        df_train      = df_train_raw.tail(cfg.bt_window)
        data          = DataHandler(cfg=cfg, df=df_train, log=cfg.log_prices)
        #exit()

        # Find the co-integration model
        model = CointegrationModel(cfg=cfg)
        res = model.fit(data.df)
        confidence_metrics = model.get_model_confidence(res)
        # Merge them for the report
        res.update({f"CONF_{k}": v for k, v in confidence_metrics.items()})
        summary_df = pd.DataFrame.from_dict(res, orient='index', columns=['Value'])
        print(summary_df)

        # Initial state from VECM
        beta_init = res["VECM_beta"][1:]
        alpha_init = res["VECM_alpha"]

        # Build y / X from DataHandler
        y = data.df[cfg.dep_col].values
        X = add_constant(data.df[cfg.indep_cols], prepend=False).values

        # MLE for noise hyperparameters
        mle = KalmanMLE(init_beta=beta_init, init_alpha=alpha_init, cfg=cfg)
        if cfg.dist_shape == 'student':
            opt_W_beta, opt_W_alpha, opt_V, opt_nu = mle.fit(y, X)
        else:
            opt_W_beta, opt_W_alpha, opt_V = mle.fit(y, X)
        
        # Fit filter
        kf = KalmanModel(W_diag=[opt_W_beta]*(cfg.state_dim-1) + [opt_W_alpha],
                        V=opt_V,
                        cfg=cfg)
        kf.initialise(beta_init, alpha_init)

        if cfg.dist_shape == 'student':
            kf.nu = opt_nu
        else:
            kf.nu = None

        df_history = kf.run_and_report(
            y, X, 
            burn_in=cfg.kalman_burn_in, 
            nu=kf.nu, 
            labels=[*cfg.indep_cols, 'Intercept'],
            dates=data.df.index
        )

        e_s = df_history["spread"].values
        Q_s = df_history["variance_Q"].values
        logging.debug(f"Mean Q:        {Q_s.mean():.6f}")
        logging.debug(f"Mean e²:       {(e_s**2).mean():.6f}")
        logging.info(f"Ratio e²/Q:    {(e_s**2).mean() / Q_s.mean():.4f}")

        df_history["future_spread"] = df_history["spread"].shift(-1) - df_history["spread"]
        # correlation
        logging.debug(df_history[["z_score", "future_spread"]].corr())

        logging.info(f"\nTraining history:")
        logging.info(df_history)
        df_history.to_csv(cfg.hist_file)

        kf.last_date = data.df.index[-1]
        kf.save_state(cfg.state_file)
        logging.info(f"Filter state saved to {cfg.state_file}. Last date: {kf.last_date}")


        # ------------------------------------------------------------------
        # WALK-FORWARD VALIDATION
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION")
        print("=" * 60)

        opt_nu_val = opt_nu if cfg.dist_shape == 'student' else None
        wfv = WalkForwardValidator(
            data      = data_raw,
            cfg       = cfg,
            beta_init = beta_init,
            alpha_init= alpha_init,
            opt_W_beta = opt_W_beta,
            opt_W_alpha= opt_W_alpha,
            opt_V      = opt_V,
            opt_nu     = opt_nu_val,
        )
        wf_summary = wfv.run()
        wf_summary.to_csv(os.path.join(cfg.output_dir, "walk_forward_summary.csv"), index=False)
        print(f"\nWalk-forward summary saved → {cfg.output_dir}walk_forward_summary.csv")
        print(f"Mean WF Sharpe: {wf_summary['Sharpe'].mean():.4f} "
              f"± {wf_summary['Sharpe'].std():.4f}")

        # ------------------------------------------------------------------
        # BACKTEST + RF + XGB TRAINING
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("BACKTEST + RF + XGB TRAINING")
        print("=" * 60)

        z_score_train = df_history["z_score"].dropna()
        entry_z = np.percentile(np.abs(z_score_train), cfg.entry_z_percentile)
        kf.entry_z = entry_z   # attach to kf before saving
        kf.exit_z  = cfg.exit_z
        kf.save_state(cfg.state_file)
 
        # Fresh filter instance for backtest (same hyperparams, re-initialised)
        kf_bt = KalmanModel(W_diag=[opt_W_beta]*(cfg.state_dim-1) + [opt_W_alpha],
                            V=opt_V,
                            cfg=cfg)
        kf_bt.nu = opt_nu if cfg.dist_shape == 'student' else None

        # Instantiate Koopman filter — runs upstream during the backtest
        koopman = KoopmanRegimeFilter(cfg=cfg)
 
        # Use data_raw to backtest on the entire training set
        # No ML gates here: we need clean labels first
        bt = Backtester(data=data_raw, kf=kf_bt, cfg=cfg, entry_z=entry_z)
        bt_results = bt.run(koopman_filter=koopman)
        bt_results.to_csv(cfg.bt_file)
 
        logging.debug(f"\nBacktest rows: {len(bt_results)}")

        num_signal = (bt_results[cfg.SIGNAL]!=0).sum()
        logging.debug(f"num_signal: {num_signal}")
        logging.info(f"% triggering: {num_signal/len(bt_results)}")
 
        # Train performance (raw, pre-filter)
        perf = Performance(cfg=cfg)
        perf.evaluate(results=bt_results, label="train_raw")
        perf.plot(results=bt_results, label="train_raw")
    
        #Build labels and feature matrix
        labels   = bt.build_labels()
        print(f"Label value counts: {labels.value_counts(dropna=False).to_dict()}")
        print(f"Signal value counts: {bt_results[cfg.SIGNAL].value_counts().to_dict()}")    
        features = bt.get_feature_matrix()
        print(f"\nLabelled trades: {labels.notna().sum()}  "
              f"(profitable: {int(labels.sum())}  "
              f"not: {int((labels == 0).sum())})")
 
        # --- Random Forest ---
        print("\n--- RF CROSS-VALIDATION ---")
        qm = SignalQualityModel(n_estimators=cfg.rf_n_estimators, cfg=cfg)
        qm.train(features, labels)
        qm.save(cfg.qm_file)
        print(f"RF model saved → {cfg.qm_file}")

        # --- XGBoost ---
        print("\n--- XGB CROSS-VALIDATION ---")
        xgb_model = XGBoostQualityModel(cfg=cfg)
        xgb_model.train(features, labels)
        xgb_model.save(cfg.xgb_file)
        print(f"XGB model saved → {cfg.xgb_file}")

        # Apply both filters and save
        filtered_train = bt.apply_quality_filter(qm=qm, xgb_model=xgb_model)
        filtered_train.to_csv(cfg.filtered_file)

        raw_sig      = bt_results[cfg.SIGNAL]
        filtered_sig = filtered_train[cfg.SIGNAL]
        logging.info(f"Train: {(raw_sig != 0).sum()} raw signals, {(filtered_sig != 0).sum()} after ML filter")
 
        # Filtered performance
        print("\n--- PERFORMANCE: RAW vs RF+XGB FILTERED ---")
        perf.compare(bt=bt, qm=qm, xgb_model=xgb_model)

        # RF-only vs XGB-only comparison for reference
        print("\n--- RF-only filter ---")
        filtered_rf_only  = bt.apply_quality_filter(qm=qm)
        perf.evaluate(results=filtered_rf_only,  label="train_rf_only")

        print("\n--- XGB-only filter ---")
        filtered_xgb_only = bt.apply_quality_filter(xgb_model=xgb_model)
        perf.evaluate(results=filtered_xgb_only, label="train_xgb_only")

        # Save the updated config
        cfg.save(cfg.config_file)
        logging.info(f"Saved updated config to {cfg.config_file}")
    else:
        # ------------------------------------------------------------------
        # PRODUCTION MODE — update state with new rows
        # ------------------------------------------------------------------

        print("=" * 60)
        print("PRODUCTION MODE")
        print("=" * 60)

        if Path(cfg.config_file).exists():
            cfg = Config.load(cfg.config_file)
            logging.info(f"Loaded config from {cfg.config_file}")
        else:
            logging.error("Couldn't load config file!")
            exit()

        bt_results = pd.read_csv(cfg.bt_file, index_col=cfg.datetime_col, parse_dates=True)
        filtered_train = pd.read_csv(cfg.filtered_file, index_col=cfg.datetime_col, parse_dates=True)
        # Check what happens after a +1 signal specifically
        long_entries  = bt_results[bt_results[cfg.SIGNAL] == 1]
        short_entries = bt_results[bt_results[cfg.SIGNAL] == -1]

        for H in [1, 2, 3, 5]:
            fwd = bt_results[cfg.PNL_RETURN].rolling(H).sum().shift(-H)
            print(f"H={H} | long mean fwd pnl: {fwd[long_entries.index].mean():.6f} "
                f"| short mean fwd pnl: {fwd[short_entries.index].mean():.6f}")

        # Also check raw z vs next return directly, bypassing signal
        logging.debug(f"\nDirect z_score vs fwd pnl:")
        for H in [1, 2, 3, 5]:
            fwd = bt_results[cfg.PNL_RETURN].rolling(H).sum().shift(-H)
            corr = bt_results[cfg.Z_SCORE].corr(fwd)
            logging.debug(f"H={H}: z_score vs fwd_{H}bar_pnl corr = {corr:.6f}")

        df_history = pd.read_csv(cfg.hist_file, index_col=cfg.datetime_col, parse_dates=True)
        e_s = df_history["spread"].values
        Q_s = df_history["variance_Q"].values
        logging.debug(f"Mean Q:        {Q_s.mean():.6f}")
        logging.debug(f"Mean e²:       {(e_s**2).mean():.6f}")
        logging.info(f"Ratio e²/Q:    {(e_s**2).mean() / Q_s.mean():.4f}")

        qm = None
        if os.path.exists(cfg.qm_file):
            qm = SignalQualityModel.load(cfg=cfg)
            print(f"RF model loaded (threshold={qm.threshold:.2f})")
        else:
            print("No RF model found -- running without RF quality filter.")

        xgb_model = None
        if os.path.exists(cfg.xgb_file):
            xgb_model = XGBoostQualityModel.load(cfg=cfg)
            print(f"XGB model loaded (threshold={xgb_model.threshold:.2f})")
        else:
            print("No XGB model found — running without XGB quality filter.")

        # Koopman filter: always instantiate fresh (stateless, runs on live data)
        koopman = KoopmanRegimeFilter(cfg=cfg)
        print(f"Koopman filter instantiated "
              f"(window={cfg.koopman_window}, n_obs={cfg.koopman_n_obs}, "
              f"eig_thresh={cfg.koopman_eig_thresh})")

        test_data = DataHandler(cfg=cfg, df=df_train_raw, df2=df_test, log=cfg.log_prices)
        test_portfolio = Portfolio(cfg=cfg)
        
        # replay=True: re-runs the full test window without advancing checkpoint
        prod_results = run_production(
            data=test_data, cfg=cfg,
            replay=cfg.replay   # don't advance checkpoint during testing
        )

        if cfg.replay:
            bt_results = pd.read_csv(cfg.bt_file, index_col=cfg.datetime_col, parse_dates=True)
            bt_results["year"] = bt_results.index.year
            yearly = bt_results.groupby("year").apply(lambda g: pd.Series({
                "sharpe":    g[cfg.PNL_RETURN].mean() / g[cfg.PNL_RETURN].std() * np.sqrt(252) if g[cfg.PNL_RETURN].std() > 0 else 0,
                "total_ret": g[cfg.PNL_RETURN].sum(),
                "win_rate":  (g[cfg.PNL_RETURN][g[cfg.SIGNAL] != 0] > 0).mean(),
                "n_trades":  (g[cfg.SIGNAL].diff().abs() > 0).sum(),
                "pct_short": (g[cfg.SIGNAL] == -1).mean(),
                "pct_long":  (g[cfg.SIGNAL] == 1).mean(),
            }))
            logging.debug(yearly.round(4))

            if not prod_results.empty:
                print("\nTest period performance")
                perf = Performance(cfg=cfg)
                perf.evaluate(results=prod_results, label="test")
                perf.plot(results=prod_results, label="test")
                perf.compare_df(bt_results, filtered_train, prod_results, 
                                    labels=["train_raw", "train_filtered", "test_filtered"])