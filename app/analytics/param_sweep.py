import dataclasses
import itertools
import pandas as pd
from app.strategies.cointegration_kalman import (
    Config, DataHandler, KalmanModel, Backtester, Performance,
)


class ParamSweepRunner:
    """Reruns Backtester/Performance with one or more Config fields
    overridden via dataclasses.replace."""

    def __init__(self, data: DataHandler, base_cfg: Config,
                 beta_init, alpha_init, w_beta: float, w_alpha: float, v: float):
        self.data, self.base_cfg = data, base_cfg
        self.beta_init, self.alpha_init = beta_init, alpha_init
        self.w_beta, self.w_alpha, self.v = w_beta, w_alpha, v

    def _run_one(self, cfg: Config, entry_z: float) -> dict:
        kf = KalmanModel(W_diag=[self.w_beta] * (cfg.state_dim - 1) + [self.w_alpha],
                          V=self.v, cfg=cfg)
        kf.initialise(self.beta_init, self.alpha_init)
        results = Backtester(data=self.data, kf=kf, cfg=cfg, entry_z=entry_z).run()
        return Performance(cfg=cfg).evaluate(results, label="sweep")

    def sweep_single(self, field: str, values: list, entry_z: float = 1.0) -> pd.DataFrame:
        rows = []
        for val in values:
            cfg = dataclasses.replace(self.base_cfg, **{field: val})
            try:
                rows.append({field: val, **self._run_one(cfg, entry_z)})
            except Exception as exc:
                rows.append({field: val, "error": str(exc)})
        return pd.DataFrame(rows)

    def sweep_grid(self, param_values: dict[str, list], entry_z: float = 1.0) -> pd.DataFrame:
        keys = list(param_values.keys())
        rows = []
        for combo in itertools.product(*param_values.values()):
            overrides = dict(zip(keys, combo))
            cfg = dataclasses.replace(self.base_cfg, **overrides)
            try:
                rows.append({**overrides, **self._run_one(cfg, entry_z)})
            except Exception as exc:
                rows.append({**overrides, "error": str(exc)})
        return pd.DataFrame(rows)