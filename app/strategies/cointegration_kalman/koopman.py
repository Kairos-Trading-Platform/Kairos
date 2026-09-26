import numpy as np
from .config import Config

class KoopmanRegimeFilter:
    """
    Extended Dynamic Mode Decomposition (EDMD) regime gate.

    Fits a linear Koopman operator on a rolling window of the spread using
    a delay-embedding observable dictionary.  The leading eigenvalue's
    modulus encodes the current mean-reversion rate:

      |lambda_1| << 1  →  fast reversion  → green light to trade
      |lambda_1| ≈ 1   →  spread near unit root  → block entry

    Architecture role: upstream gate, called BEFORE the Kalman filter
    drives signal generation.  If is_stationary() returns False, the
    calling code should skip signal generation entirely for that bar.

    Parameters (in Config)
    ----------------------
    koopman_window     : int   number of observations used per EDMD fit
    koopman_n_obs      : int   number of delay embeddings (dictionary size)
    koopman_eig_thresh : float leading |eigenvalue| ceiling (default 0.97)
    """

    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.window     = cfg.koopman_window
        self.n_obs      = cfg.koopman_n_obs
        self.thresh     = cfg.koopman_eig_thresh
        self._lead_eig  = np.nan          # last estimated leading |eigenvalue|
        self._eig_hist  = []              # rolling history for diagnostics

    # ------------------------------------------------------------------
    # Core EDMD
    # ------------------------------------------------------------------
    def _build_psi(self, x: np.ndarray) -> np.ndarray:
        """
        Build the observable matrix Psi using a delay-embedding dictionary.

        For a 1-D signal x of length T, we create a (T - n_obs) × n_obs
        matrix where row t contains [x_t, x_{t-1}, ..., x_{t-n_obs+1}].
        This is the Hankel / time-delay embedding, a standard Koopman dict
        for scalar time series.
        """
        T = len(x)
        if T <= self.n_obs:
            return None
        rows = T - self.n_obs
        Psi  = np.column_stack([x[i : i + rows] for i in range(self.n_obs - 1, -1, -1)])
        return Psi   # shape (rows, n_obs)

    def _fit_koopman(self, spread_window: np.ndarray) -> float:
        """
        Solve the EDMD least-squares problem:  Psi_prime ≈ K @ Psi.T

        Returns the modulus of the leading eigenvalue of K, or nan on failure.
        """
        Psi = self._build_psi(spread_window)
        if Psi is None or Psi.shape[0] < self.n_obs + 2:
            return np.nan

        Psi_x  = Psi[:-1]   # "current"  observables  shape (M, n_obs)
        Psi_y  = Psi[1:]    # "next-step" observables  shape (M, n_obs)

        # K = argmin ||Psi_y - Psi_x @ K||_F   (transpose convention)
        # Solved via pseudo-inverse: K = pinv(Psi_x) @ Psi_y
        try:
            K      = np.linalg.lstsq(Psi_x, Psi_y, rcond=None)[0]  # (n_obs, n_obs)
            eigvals = np.linalg.eigvals(K)
            lead    = float(np.max(np.abs(eigvals)))
            return lead
        except np.linalg.LinAlgError:
            return np.nan

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, spread_buffer: np.ndarray) -> float:
        """
        Update the Koopman estimate from the latest spread buffer.

        Parameters
        ----------
        spread_buffer : array of recent raw spread values (innovations e_t),
                        length >= koopman_window.

        Returns
        -------
        Leading eigenvalue modulus (float), or nan if estimation failed.
        """
        if len(spread_buffer) < self.window:
            self._lead_eig = np.nan
            return np.nan

        window_data    = np.array(spread_buffer[-self.window:], dtype=float)
        # Standardise so EDMD isn't scale-sensitive
        std = window_data.std()
        if std < 1e-10:
            self._lead_eig = np.nan
            return np.nan
        window_norm    = (window_data - window_data.mean()) / std

        self._lead_eig = self._fit_koopman(window_norm)
        self._eig_hist.append(self._lead_eig)
        return self._lead_eig

    def is_stationary(self) -> bool:
        """
        Returns True if the spread is in a mean-reverting regime.

        A NaN eigenvalue (too few data) defaults to True so the filter
        doesn't block entries during warm-up.
        """
        if np.isnan(self._lead_eig):
            return True   # warm-up: don't block
        return self._lead_eig < self.thresh

    @property
    def lead_eigenvalue(self) -> float:
        return self._lead_eig

    def diagnostics(self) -> dict:
        hist = np.array([e for e in self._eig_hist if not np.isnan(e)])
        return {
            "koopman_lead_eig_mean": float(np.mean(hist)) if len(hist) else np.nan,
            "koopman_lead_eig_std":  float(np.std(hist))  if len(hist) else np.nan,
            "koopman_block_rate":    float((hist >= self.thresh).mean()) if len(hist) else np.nan,
        }
