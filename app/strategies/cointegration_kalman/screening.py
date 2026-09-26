"""Persistent, dashboard-readable record of every pair tested for
cointegration. Figures are stored as Plotly JSON so the frontend renders
them through the same Plotly.react() path used everywhere else here."""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .exceptions import DiagnosticFailure


class ScreeningLog:
    """Appends one JSON record per tested pair to <root>/screening_log.jsonl."""

    def __init__(self, root_dir: str = "output/"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root_dir / "screening_log.jsonl"
        self.last_plot_json: str | None = None  # set on record_failure, for immediate UI feedback

    def record_pass(self, dep_col: str, indep_col: str, summary: dict, pair_output_dir: str):
        self.last_plot_json = None
        self._append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dep_col": dep_col, "indep_col": indep_col,
            "status": "passed", "summary": summary,
            "output_dir": pair_output_dir,
        })

    def record_failure(self, dep_col: str, indep_col: str, exc: Exception, pair_output_dir: str):
        if isinstance(exc, DiagnosticFailure):
            stage, reason, stats = exc.stage, exc.reason, exc.stats
            self.last_plot_json = self._build_plot(dep_col, indep_col, exc)
        else:
            stage, reason, stats = "unexpected_error", str(exc), {}
            self.last_plot_json = None

        self._append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dep_col": dep_col, "indep_col": indep_col,
            "status": "failed", "stage": stage, "reason": reason,
            "stats": stats, "plot": self.last_plot_json,
            "output_dir": pair_output_dir,
        })

    def load(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def _append(self, record: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _build_plot(self, dep_col: str, indep_col: str, exc: DiagnosticFailure) -> str | None:
        """One subplot per series attached to the exception — proves the
        rejection (e.g. a flat/mean-reverting level series disproving I(1))."""
        if not exc.series:
            return None

        ROLLING_SUFFIXES = ("_rolling_adf_stat", "_rolling_adf_crit_5pct")

        raw_panels: dict[str, pd.Series] = {}
        rolling_groups: dict[str, dict[str, pd.Series]] = {}

        for name, s in exc.series.items():
            matched_suffix = next((suf for suf in ROLLING_SUFFIXES if name.endswith(suf)), None)
            if matched_suffix:
                base = name[: -len(matched_suffix)]
                rolling_groups.setdefault(base, {})[name] = s
            else:
                raw_panels[name] = s

        panel_order = list(raw_panels.keys()) + list(rolling_groups.keys())
        titles = list(raw_panels.keys()) + [f"{b} — rolling ADF vs 5% crit" for b in rolling_groups.keys()]

        fig = make_subplots(rows=len(panel_order), cols=1,
                            subplot_titles=titles, vertical_spacing=0.08)

        row = 1
        for name, s in raw_panels.items():
            s = pd.Series(s).dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name), row=row, col=1)
            fig.add_hline(y=float(s.mean()), line_dash="dash", line_color="gray", row=row, col=1)
            row += 1

        for base, members in rolling_groups.items():
            for name, s in members.items():
                s = pd.Series(s).dropna()
                is_crit = name.endswith("_crit_5pct")
                fig.add_trace(go.Scatter(
                    x=s.index, y=s.values, mode="lines", name=name,
                    line=dict(dash="dot" if is_crit else "solid", color="red" if is_crit else None),
                ), row=row, col=1)
            row += 1

        fig.update_layout(
            title=f"{dep_col} vs {indep_col} — {exc.stage} failed: {exc.reason}",
            height=320 * len(panel_order), showlegend=True,
        )
        return fig.to_json()