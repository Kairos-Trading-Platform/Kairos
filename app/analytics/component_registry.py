from __future__ import annotations
import dataclasses
from app.strategies.cointegration_kalman import (
    Config, CointegrationModel, KalmanModel, KalmanMLE, SignalEngine,
    Portfolio, RandomForestModel, XGBoostModel, KoopmanRegimeFilter,
    Backtester, WalkForwardValidator,
)


@dataclasses.dataclass(frozen=True)
class ComponentSpec:
    key: str
    label: str
    cls: type
    config_fields: tuple[str, ...]


class ComponentRegistry:
    """Single source of truth: which Config fields belong to which class.
    Param types/defaults are introspected from Config itself — never
    hand-duplicated here."""

    _registry: dict[str, ComponentSpec] = {}

    @classmethod
    def register(cls, key, label, target_cls, fields):
        cls._registry[key] = ComponentSpec(key, label, target_cls, tuple(fields))

    @classmethod
    def list(cls) -> list[dict]:
        return [{"key": s.key, "label": s.label} for s in cls._registry.values()]

    @classmethod
    def schema(cls, key: str) -> list[dict]:
        spec = cls._registry[key]
        field_map = {f.name: f for f in dataclasses.fields(Config)}
        out = []
        for name in spec.config_fields:
            f = field_map[name]
            default = f.default if f.default is not dataclasses.MISSING else None
            out.append({"key": name, "type": type(default).__name__, "default": default})
        return out


ComponentRegistry.register("cointegration", "Cointegration Model (VECM)", CointegrationModel,
    ["asset_class", "coint_maxlags", "coint_min_window", "coint_enforce_i1",
     "coint_enforce_po", "coint_enforce_johansen", "coint_confirm_n", "coint_confirm_frac"])

ComponentRegistry.register("kalman", "Kalman Filter", KalmanModel,
    ["kalman_burn_in_raw", "kalman_norm_window", "kalman_C_prior",
     "kalman_W_default_beta", "kalman_W_default_alpha", "kalman_V_default", "kalman_debug"])

ComponentRegistry.register("kalman_mle", "Kalman MLE", KalmanMLE,
    ["dist_shape", "mle_W_beta_lo", "mle_W_beta_hi", "mle_W_alpha_lo", "mle_W_alpha_hi",
     "mle_V_lo", "mle_V_hi", "mle_n_starts", "mle_nu_lo", "mle_nu_hi", "mle_fix_V_from_ols"])

ComponentRegistry.register("signal_engine", "Signal Engine", SignalEngine,
    ["entry_z_percentile", "exit_z", "z_score_std_tol", "z_score_std_conf",
     "stop_loss_mult", "regime_window", "regime_sigma", "regime_filter", "min_hold_frac"])

ComponentRegistry.register("portfolio", "Portfolio Sizing", Portfolio,
    ["capital", "target_vol", "cost_bps"])

ComponentRegistry.register("random_forest", "Random Forest Gate", RandomForestModel,
    ["rf_n_estimators", "rf_n_splits", "rf_threshold"])

ComponentRegistry.register("xgboost", "XGBoost Gate", XGBoostModel,
    ["xgb_n_estimators", "xgb_max_depth", "xgb_learning_rate", "xgb_subsample", "xgb_threshold"])

ComponentRegistry.register("koopman", "Koopman Regime Filter", KoopmanRegimeFilter,
    ["koopman_window", "koopman_n_obs", "koopman_eig_thresh"])

ComponentRegistry.register("backtester", "Backtester", Backtester,
    ["bt_window", "label_horizon"])

ComponentRegistry.register("walk_forward", "Walk-Forward Validator", WalkForwardValidator,
    ["wf_n_splits", "wf_min_train_frac"])