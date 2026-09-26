from dataclasses import dataclass, field
import dataclasses
from typing import List
import logging # TODO logger
import os
from scipy.stats import chi2
import numpy as np
import json
from pathlib import Path

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
