import numpy as np
from .config import Config

class SignalEngine:
    """
    Mean-reversion signal from z-score with hysteresis and regime warning.
    Regime warning uses a rolling baseline of |delta_alpha| so the threshold
    adapts to the filter's typical daily adjustment rather than a fixed value.
    """

    def __init__(self, entry_z: float = 1.0, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to SignalEngine.")
        self.cfg = cfg
        self.entry_z              = entry_z
        self.exit_z               = abs(cfg.exit_z)
        self.regime_window        = cfg.regime_window
        self.regime_sigma         = cfg.regime_sigma
        self.regime_filter        = cfg.regime_filter
        self.stop_loss_multiplier = cfg.stop_loss_mult
        self.min_hold_frac        = cfg.min_hold_frac
        self.position             = 0
        self.bars_in_trade        = 0
        self._alpha_history       = []

    def generate(self, z: float, delta_alpha: float = 0.0, half_life: float = 20.0):
        """
        Returns (position, regime_warning).
 
        regime_warning is True when |delta_alpha| exceeds the rolling mean
        + regime_sigma * rolling_std of recent alpha changes.
        """
        self._alpha_history.append(abs(delta_alpha))
        buf = self._alpha_history[-self.regime_window:]

        # Regime detection
        if len(buf) >= 5:
            mu, sigma      = np.mean(buf), np.std(buf)
            regime_warning = abs(delta_alpha) > mu + self.regime_sigma * max(sigma, 1e-4)
        else:
            regime_warning = False

        stop = self.stop_loss_multiplier * self.entry_z

        # Entry logic
        if self.position == 0:
            self.bars_in_trade = 0 # Reset counter
            # Block new entries during regime shifts if filter is on
            if self.regime_filter and regime_warning:
                pass   # Stay flat
            elif z > self.entry_z: self.position = -1 # Spread too high -> short
            elif z < -self.entry_z: self.position = +1 # spread too low -> long
        # Exit logic
        else:
            self.bars_in_trade += 1
            # Target (hysteresis) — only allowed after minimum holding period
            min_hold_bars = max(1, int(self.min_hold_frac * half_life))
            held_long_enough = self.bars_in_trade >= min_hold_bars
            reached_target = held_long_enough and (
                (self.position == +1 and z > -self.exit_z) or
                (self.position == -1 and z < self.exit_z)
            )
            # Time decay — always allowed (prevents stale positions)
            time_decay = self.bars_in_trade > (2.0 * half_life)
            # Stop-loss — always allowed (risk management must never be blocked)
            hit_stop = (self.position == +1 and z < -stop) or \
                       (self.position == -1 and z > stop)
            if reached_target or time_decay or hit_stop:
                self.position = 0
        return self.position, regime_warning
