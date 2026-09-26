import numpy as np
import pandas as pd
from .config import Config

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
   