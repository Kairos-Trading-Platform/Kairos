from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class StrategyParam:
    key: str
    label: str
    type: str              # 'ticker' | 'multi_ticker' | 'int' | 'float' | 'select'
    default: object
    options: list = field(default_factory=list)

@dataclass
class StrategyResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    signal: pd.Series | None = None
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # weights, regime labels, etc.

class BaseStrategy(ABC):
    key: str
    label: str

    @abstractmethod
    def param_schema(self) -> list[StrategyParam]: ...

    @abstractmethod
    def run(self, price_history: pd.DataFrame, params: dict) -> StrategyResult: ...

class StrategyRegistry:
    _registry: dict[str, type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_cls):
        cls._registry[strategy_cls.key] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, key: str) -> BaseStrategy:
        return cls._registry[key]()

    @classmethod
    def list(cls) -> list[dict]:
        return [{"key": k, "label": v.label} for k, v in cls._registry.items()]