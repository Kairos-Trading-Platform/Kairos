import numpy as np
from .config import Config

class Portfolio:
    """
    Volatility-targeted dollar-neutral portfolio.

    At each signal entry:
      - Allocates `capital` across legs using hedge-ratio weights
      - Scales position size so expected daily spread vol = target_vol
      - Models transaction costs as a fixed fraction of notional traded
        (not of PnL), applied at entry and exit

    Attributes
    ----------
    capital    : total capital in currency units (e.g. USD)
    target_vol : target daily portfolio volatility as a fraction (e.g. 0.01 = 1%)
    cost_bps   : one-way transaction cost in basis points (e.g. 5 = 0.05%)
    """
    def __init__(self, cfg: Config = None,
                 capital: float = None,
                 target_vol: float = None,
                 cost_bps: float = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to Portfolio.")
        self.cfg = cfg
        self.capital   = capital   or cfg.capital
        self.target_vol = target_vol or cfg.target_vol
        self.cost_bps  = cost_bps  or cfg.cost_bps
        self.records   = []
        self.position  = 0
        self.last_cost = 0.0
        self.last_weights = None


    def compute_weights(self, beta: np.ndarray,
                        spread_vol: float) -> np.ndarray:
        """
        Returns weight vector w = [w_dep, w_indep_1, ..., w_indep_n].

        Step 1 — hedge ratio: long $1 of dependent asset, short $beta of independent assets
        Step 2 — normalise: sum(|w|) = 1  → weights are fractions of capital
        Step 3 — vol scale: multiply by target_vol / spread_vol so that a
                 1-sigma spread move produces target_vol return on capital
        """
        w     = np.empty(len(beta) + 1)
        w[0]  = 1.0
        w[1:] = -beta
        w    /= np.abs(w).sum()                          # step 2
        if spread_vol > 1e-10:
            w *= self.target_vol / spread_vol            # step 3
        return w

    def compute_shares(self, weights: np.ndarray,
                       prices: np.ndarray) -> np.ndarray:
        """
        Converts weight vector to number of shares per leg.

        shares_i = capital * weight_i / price_i

        Positive = long, negative = short.
        """
        notional = self.capital * weights                # dollars per leg
        return notional / prices                         # shares per leg

    def notional_traded(self, weights: np.ndarray) -> float:
        """Total gross notional = sum of absolute dollar values per leg."""
        return float(np.abs(weights).sum() * self.capital)

    def transaction_cost(self, weights: np.ndarray) -> float:
        """
        One-way cost = cost_bps/10000 * gross notional.
        Applied once at entry and once at exit.
        """
        return (self.cost_bps / 10_000) * self.notional_traded(weights)

    def update(self, log_returns: np.ndarray,
               weights: np.ndarray,
               signal: int,
               prices: np.ndarray,
               prev_signal: int) -> dict:
        """
        Computes PnL and costs for one bar.

        Parameters
        ----------
        log_returns : log-return of each asset this bar
        weights     : from compute_weights()
        signal      : current position (-1, 0, +1)
        prices      : current raw prices for share calculation
        prev_signal : signal from the previous bar (to detect transitions)

        Returns a dict with all metrics for this bar.
        """
        # P&L in return units and dollars
        pnl_return = float(signal * np.dot(weights, log_returns))
        pnl_dollar = pnl_return * self.capital

        if signal == 0:
            assert pnl_dollar == 0.0, f"pnl_dollar not 0"

        # Transaction costs: charged on entry and exit transitions
        cost = 0.0
        entered = (prev_signal == 0 and signal != 0)
        exited  = (prev_signal != 0 and signal == 0)
        # We use 'weights' from the current bar for entry, 
        # but for exit, we must use weights from the PREVIOUS bar (what we actually hold).
        if entered:
            cost = self.transaction_cost(weights)
            self.last_weights = weights.copy()
        elif exited:
            cost = self.transaction_cost(self.last_weights) if self.last_weights is not None else 0.0
        
        # Share breakdown per leg
        is_active = (signal != 0)
        shares = self.compute_shares(weights * signal, prices)

        record = {
            self.cfg.PNL_RETURN: (pnl_dollar - cost) / self.capital,
            self.cfg.PNL_DOLLAR: pnl_dollar - cost,
            self.cfg.PNL_GROSS:  pnl_dollar,
            self.cfg.COST: cost,
            self.cfg.NOTIONAL: self.notional_traded(weights) if is_active else 0.0,
            self.cfg.SHARES_DEP: shares[0], # Dependent shares (+ = long)
            self.cfg.SHARES_INDEP: shares[1:], # Independent shares (- = short)
            self.cfg.WEIGHTS_DEP:   weights[0] * signal,
            self.cfg.WEIGHTS_INDEP: weights[1:] * signal,
            self.cfg.ENTRY_PRICE: prices[0] if entered else np.nan,
            self.cfg.EXIT_PRICE: prices[0] if exited  else np.nan,
        }

        self.records.append(record)
        self.last_cost = cost
        return record
