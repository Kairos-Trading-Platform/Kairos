"""Structured failure signalling for pair screening / research diagnostics."""
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd


@dataclass
class DiagnosticFailure(Exception):
    """
    Raised instead of a bare ValueError whenever a pair fails a statistical
    pre-check (integration order, PO test, Johansen, ...). Carries the
    numbers and series already computed at the point of failure so the
    dashboard can render a proof plot without recomputing anything.
    """
    stage: str
    reason: str
    stats: Dict[str, float] = field(default_factory=dict)
    series: Dict[str, pd.Series] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.reason