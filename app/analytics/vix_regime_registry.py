from app.strategies.base import BaseStrategy, StrategyParam, StrategyResult, StrategyRegistry
from app.strategies.vix_regime_modeling import (
    MarketData, GaussianHMM_VIX, WalkForwardRegimeModel, BacktestEngine
)

@StrategyRegistry.register
class VixRegimeStrategy(BaseStrategy):
    key, label = "vix_regime", "VIX Regime Rotation (HMM)"

    def param_schema(self):
        return [
            StrategyParam("tickers", "Rotation universe", "multi_ticker", ["TLT", "GLD", "SPY"]),
            StrategyParam("n_states", "Number of regimes", "select", 3, options=[2, 3]),
            StrategyParam("refit_freq", "Refit frequency", "select", "3ME", options=["1ME", "3ME", "6ME"]),
        ]

    def run(self, price_history, params) -> StrategyResult:
        md = MarketData(tickers=params["tickers"]).run()  # or inject price_history directly
        model = WalkForwardRegimeModel(
            model_factory=lambda: GaussianHMM_VIX(n_states=params.get("n_states", 3)),
            dvix=md.dvix, refit_freq=params.get("refit_freq", "3ME"),
        )
        engine = BacktestEngine(returns=md.returns, dvix_index=md.dvix.index)
        engine.add_model_strategy(self.key, model)
        rets = engine.strategies_returns[self.key]

        return StrategyResult(
            equity_curve=(1 + rets).cumprod() - 1,
            daily_returns=rets,
            metrics=engine.performance_table().loc[self.key].to_dict(),
        )