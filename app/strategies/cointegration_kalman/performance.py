import logging # TODO logger
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from .backtester import Backtester
from .config import Config

class Performance:
    """Strategy evaluation: Sharpe, Calmar, max drawdown, turnover, win rate."""
    def __init__(self, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to Performance.")
        self.cfg = cfg

    def evaluate(self, results: pd.DataFrame, label: str = "") -> dict:
        pnl      = results[self.cfg.PNL_RETURN]
        signal   = results[self.cfg.SIGNAL]
        cum      = pnl.cumsum()
        drawdown = cum - cum.cummax()

        # Annualised return — infer years from index if DatetimeIndex
        if isinstance(results.index, pd.DatetimeIndex):
            years = (results.index[-1] - results.index[0]).days / 365.25
        else:
            years = len(results) / 252   # fallback: assume daily
        ann_return = cum.iloc[-1] / years if years > 0 else np.nan

        turnover = (signal.diff().abs() > 0).mean()
        sharpe   = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
        max_dd   = drawdown.min()
        calmar   = ann_return / abs(max_dd) if max_dd != 0 else np.nan
        active   = pnl[signal != 0]
 
        metrics = {
            "Sharpe":      sharpe,
            "Calmar":      calmar,
            "MaxDrawdown": max_dd,
            "TotalReturn": cum.iloc[-1],
            "AnnReturn":  ann_return,
            "Turnover":    turnover,
            "WinRate":     (active > 0).mean() if len(active) else np.nan,
            "NumTrades":   int((signal.diff().abs() > 0).sum()),
        }
 
        tag = f" [{label}]" if label else ""
        print(f"\nPerformance{tag}")
        for k, v in metrics.items():
            logging.info(f"  {k:<15} {v:.4f}")
 
        plt.figure(figsize=(12, 4))
        cum.plot(title=f"Cumulative PnL{tag}", grid=True)
        plt.tight_layout()
        fname = f"equity_curve{'_' + label if label else ''}.png"
        path = os.path.join(self.cfg.output_dir, fname)
        plt.savefig(path)
        plt.close()
 
        return metrics

    def compare(self, bt: 'Backtester', qm=None, xgb_model=None):
        """Side-by-side Sharpe / MaxDD before and after ML gating."""
        raw      = bt.results
        filtered = bt.apply_quality_filter(qm=qm, xgb_model=xgb_model)
        print("\n=== Before ML filter ===")
        self.evaluate(raw, label="raw")
        print("\n=== After ML filter ===")
        self.evaluate(filtered, label="filtered")
        

    def compare_df(self, *dfs: pd.DataFrame, labels: list = None):
        # Set default labels if not provided
        labels = labels or [f"df{i+1}" for i in range(len(dfs))]
        # Evaluate performance for each DataFrame
        metrics = [self.evaluate(df, label=lbl) for df, lbl in zip(dfs, labels)]
        rows = ["Sharpe", "Calmar", "MaxDrawdown", "TotalReturn",
                "AnnReturn", "WinRate", "Turnover", "NumTrades"]
        # Header
        header = f"{'Metric':<15}" + "".join(f"{lbl:>14}" for lbl in labels)
        print(f"\n{header}")
        print("-" * (15 + 14 * len(labels)))
        # Print metrics for each row
        for row in rows:
            vals = "".join(f"{m.get(row, float('nan')):>14.4f}" for m in metrics)
            logging.info(f"{row:<15}{vals}")

        # Plot and analyze each DataFrame
        for df, lbl in zip(dfs, labels):
            self.plot_z_linearity(df, label=lbl)
            self.calculate_ic_decay(df)
            self.run_shuffled_test(df, label=lbl)
            
    def plot(self, results: pd.DataFrame, label: str = ""):
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        pnl      = results[self.cfg.PNL_RETURN]
        cum      = pnl.cumsum()
        drawdown = cum - cum.cummax()
        signal   = results[self.cfg.SIGNAL]
        rolling_sharpe = (pnl.rolling(63).mean()
                        / pnl.rolling(63).std() * np.sqrt(252))

        # 1. Cumulative PnL
        axes[0].plot(cum.index, cum.values)
        axes[0].set_title(f"Cumulative PnL [{label}]")
        axes[0].set_ylabel("Cumulative log-return")
        axes[0].grid(True, alpha=0.3)

        # 2. Drawdown
        axes[1].fill_between(drawdown.index, drawdown.values, 0,
                            color="red", alpha=0.4)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        # 3. Rolling 63-bar Sharpe
        axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, color="green")
        axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[2].set_title("Rolling Sharpe (63 bars)")
        axes[2].set_ylabel("Sharpe")
        axes[2].grid(True, alpha=0.3)

        # 4. Signal and position
        axes[3].plot(signal.index, signal.values,
                    linewidth=0.5, color="purple")
        axes[3].set_title(self.cfg.SIGNAL)
        axes[3].set_ylabel("Position")
        axes[3].set_yticks([-1, 0, 1])
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{self.cfg.output_dir}_performance_{label}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        logging.info(f"Performance plot saved to {fname}")

    def plot_z_linearity(self, results: pd.DataFrame, label: str = "", bins: int = 5):
        """
        Check if stronger Z-scores lead to higher average returns.
        Belongs in Research Diagnostics.
        """
        df = results.copy()
        # Use absolute Z-score to measure the 'strength' of the signal
        # We assume the Z-score column is available in results via config
        z_col = self.cfg.Z_SCORE # Ensure this is in your Config class
        df['abs_z'] = df[z_col].abs()
        
        # Create quintiles
        df['z_bin'] = pd.qcut(df['abs_z'], bins, 
                             labels=[f'Q{i+1}' for i in range(bins)])
        
        linearity = df.groupby('z_bin', observed=True)[self.cfg.PNL_RETURN].mean()
        
        plt.figure(figsize=(8, 5))
        linearity.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.title(f"Z-Score Linearity [{label}]")
        plt.ylabel("Mean Bar Return")
        plt.xlabel("Z-Score Magnitude Quintile")
        plt.tight_layout()
        
        fname = os.path.join(self.cfg.output_dir,f"z_linearity_{label}.png")
        plt.savefig(fname)
        plt.close()
        logging.info(f"Z-Linearity plot saved to {fname}")
        return linearity

    def calculate_ic_decay(self, results: pd.DataFrame, horizons=[1, 2, 5, 10, 20]):
        """
        Calculate Information Coefficient (Spearman Rank Correlation) across horizons.
        Measures predictive power decay.
        """
        z_col = self.cfg.Z_SCORE
        pnl_col = self.cfg.PNL_RETURN
        ic_results = {}
        
        for h in horizons:
            # We correlate current Z-score with the sum of returns over the NEXT 'h' bars
            fwd_ret = results[pnl_col].shift(-h).rolling(window=h).sum()
            
            # Spearman is preferred as it is less sensitive to outliers
            ic = results[z_col].corr(fwd_ret, method='spearman')
            ic_results[f"H={h}"] = ic
            
        print(f"\nInformation Coefficient (IC) Decay:")
        for h, val in ic_results.items():
            logging.info(f"  {h}: {val:.4f}")
            
        return ic_results

    def run_shuffled_test(self, results: pd.DataFrame, iterations: int = 100, label: str = ""):
        """
        Monte Carlo test: scramble returns to see if PnL is due to timing or just drift.
        """
        real_pnl = results[self.cfg.PNL_RETURN].values
        real_cum = np.cumsum(real_pnl)
        
        plt.figure(figsize=(10, 6))
        
        # Run iterations
        shuffled_sharpes = []
        for _ in range(iterations):
            shuffled = np.random.permutation(real_pnl)
            shuffled_cum = np.cumsum(shuffled)
            plt.plot(shuffled_cum, color='gray', alpha=0.1)
            
            # Calculate Sharpe for this shuffle
            shuffled_sharpes.append(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252))
            
        plt.plot(real_cum, color='blue', linewidth=2, label='Actual Strategy')
        plt.title(f"Shuffled Test ({iterations} iterations) [{label}]")
        plt.ylabel("Cumulative Return")
        plt.legend()
        
        fname = os.path.join(self.cfg.output_dir,f"shuffled_test_{label}.png")
        plt.savefig(fname)
        plt.close()
        
        real_sharpe = np.mean(real_pnl) / np.std(real_pnl) * np.sqrt(252)
        percentile = (np.array(shuffled_sharpes) < real_sharpe).mean()
        
        logging.info(f"\nShuffled Test [{label}]:")
        logging.info(f"  Actual Sharpe: {real_sharpe:.4f}")
        logging.info(f"  Strategy Percentile vs Random: {percentile*100:.1f}%")
