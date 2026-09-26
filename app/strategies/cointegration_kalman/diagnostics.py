from arch.unitroot import ADF
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
import logging # TODO logger

class StatisticalTestEngine:
    """Rolling ADF, half-life, and CUSUM diagnostics."""
    def compute_all(self, df: pd.DataFrame, spread: pd.Series,
                    johansen_res) -> dict:
        result = {}
 
        for key, series in [
            ("adf_level_p", df.iloc[:, 0], "n"),
            ("adf_resid_p", spread,         "n"),
        ]:
            s = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 20 or s.std() < 1e-8:
                result[key] = np.nan
            else:
                # Cap max_lags so ADF never requests more lags than the
                # series can support (rule of thumb: n/5, capped at 10)
                safe_lags = min(10, max(1, len(s) // 5))
                try:
                    result[key] = ADF(s, trend="n", method="bic",
                                      max_lags=safe_lags).pvalue
                except Exception as exc:
                    logging.error(f"  ADF failed [{key}, n={len(s)}, "
                          f"max_lags={safe_lags}]: {exc}")
                    result[key] = np.nan
 
        result["johansen_trace"] = johansen_res.trace_stat[0]
        result["half_life"]      = self._estimate_half_life(spread)
        return result

    def _estimate_half_life(self, spread: pd.Series) -> float:
        s = spread.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20:
            return np.nan
        df = pd.DataFrame({"lag": s.shift(1), "ret": s - s.shift(1)}).dropna()
        if len(df) < 20:
            return np.nan
        lam = Polynomial.fit(df["lag"], df["ret"], 1).convert().coef[1]
        return np.nan if lam >= 0 else -np.log(2) / lam


    def cusum_test(self, std_innov: np.ndarray, significance: float = 0.05,
                   burn_in: int = 0):
        n   = len(std_innov)
        eff = std_innov[burn_in:]
        n_e = len(eff)
        c   = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}[significance]
 
        cusum_val = np.cumsum(eff) / np.sqrt(n_e)
        t_i       = np.arange(1, n_e + 1)
        upper_val =  c * np.sqrt(t_i / n_e)
        lower_val = -c * np.sqrt(t_i / n_e)
 
        cusum = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        cusum[burn_in:] = cusum_val
        upper[burn_in:] = upper_val
        lower[burn_in:] = lower_val
 
        detected, in_break = [], False
        for i in range(2, n_e):
            outside = cusum[i] > upper[i] or cusum[i] < lower[i]
            if not in_break and outside:
                detected.append(i + burn_in)
                in_break = True
            elif in_break and not outside:
                in_break = False
 
        return cusum, upper, lower, detected


    def rolling_adf(self, residuals: np.ndarray, window: int = 120):
        num = len(residuals)
        adf_stats, crit_5pct = np.full(num, np.nan), np.full(num, np.nan)

        for t in range(window, num):
            try:
                # trend="n" is correct for residuals
                res = ADF(residuals[t - window : t], trend="n")
                adf_stats[t] = res.stat
                crit_5pct[t] = res.critical_values["5%"]
            except:
                continue
        return adf_stats, None, crit_5pct # keep return signature consistent
