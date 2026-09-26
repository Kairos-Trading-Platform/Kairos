import numpy as np
import logging # TODO logger
import pandas as pd
from .executor import StrategyExecutor
from .signal_engine import SignalEngine
from .portfolio import Portfolio
from .cointegration import CointegrationModel
from .diagnostics import StatisticalTestEngine
from .data_handler import DataHandler
from .config import Config
from .kalman import KalmanModel

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
