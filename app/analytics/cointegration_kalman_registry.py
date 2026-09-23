from app.strategies.base import BaseStrategy, StrategyParam, StrategyResult, StrategyRegistry
from app.strategies.cointegration_kalman_class import (
    Config, DataHandler, CointegrationModel, KalmanModel, KalmanMLE, Backtester, Performance
)

@StrategyRegistry.register
class CointegrationKalmanStrategy(BaseStrategy):
    key, label = "coint_kalman", "Cointegration Kalman Pairs"

    def param_schema(self):
        return [
            StrategyParam("dep_col", "Dependent ticker", "ticker", None),
            StrategyParam("indep_cols", "Independent ticker(s)", "multi_ticker", []),
            StrategyParam("bt_window", "Backtest window", "int", 252),
            StrategyParam("entry_z_percentile", "Entry Z percentile", "float", 90.0),
        ]

    def run(self, price_history, params) -> StrategyResult:
        cfg = Config(dep_col=params["dep_col"], indep_cols=params["indep_cols"],
                     bt_window=params.get("bt_window", 252),
                     entry_z_percentile=params.get("entry_z_percentile", 90.0))

        cols = [cfg.dep_col, *cfg.indep_cols]
        data = DataHandler(cfg=cfg, df=price_history[cols], log=cfg.log_prices)

        res = CointegrationModel(cfg).fit(data.df.tail(cfg.bt_window))
        mle = KalmanMLE(init_beta=res["VECM_beta"][1:], init_alpha=res["VECM_alpha"], cfg=cfg)
        opt = mle.fit(data.df[cfg.dep_col].values,
                       data.df[cfg.indep_cols].assign(const=1.0).values)

        kf = KalmanModel(W_diag=opt[:cfg.state_dim], V=opt[cfg.state_dim], cfg=cfg)
        kf.initialise(res["VECM_beta"][1:], res["VECM_alpha"])

        bt = Backtester(data=data, kf=kf, cfg=cfg, entry_z=params.get("entry_z_percentile", 90.0))
        results = bt.run()
        metrics = Performance(cfg).evaluate(results, label=self.key)

        return StrategyResult(
            equity_curve=results[cfg.PNL_RETURN].cumsum(),
            daily_returns=results[cfg.PNL_RETURN],
            signal=results[cfg.SIGNAL],
            metrics=metrics,
        )