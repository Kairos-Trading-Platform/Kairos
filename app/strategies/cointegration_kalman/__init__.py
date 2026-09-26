"""
app.strategies.coint_kalman

Re-exports the public API so existing callers (e.g. cointegration_kalman_registry.py,
research.py) keep working unchanged:

    from app.strategies.coint_kalman import Config, DataHandler, ...
"""
from .config import Config, StepResult
from .data_handler import DataHandler
from .cointegration import CointegrationModel
from .kalman import KalmanModel, KalmanMLE
from .diagnostics import StatisticalTestEngine
from .signal_engine import SignalEngine
from .portfolio import Portfolio
from .quality_models import QualityModelBase, RandomForestModel, XGBoostModel
from .koopman import KoopmanRegimeFilter
from .executor import StrategyExecutor
from .backtester import Backtester
from .walk_forward import WalkForwardValidator
from .performance import Performance
from .pipeline import (
    run_production,
    compute_and_save_all_pairs,
    find_first_significant_window,
    automate_strategy_params,
    download,
)

__all__ = [
    "Config", "StepResult",
    "DataHandler",
    "CointegrationModel",
    "KalmanModel", "KalmanMLE",
    "StatisticalTestEngine",
    "SignalEngine",
    "Portfolio",
    "QualityModelBase", "RandomForestModel", "XGBoostModel",
    "KoopmanRegimeFilter",
    "StrategyExecutor",
    "Backtester",
    "WalkForwardValidator",
    "Performance",
    "run_production",
    "compute_and_save_all_pairs",
    "find_first_significant_window",
    "automate_strategy_params",
    "download",
]