from arch.unitroot import ADF
import pandas as pd
import logging # TODO logger
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR
from arch.unitroot.cointegration import phillips_ouliaris
import numpy as np
from .config import Config
from .exceptions import DiagnosticFailure

class CointegrationModel:
    """
    Fits a VECM and runs Johansen + ADF diagnostics on a price window.
    Returns a structured result dict consumed by KalmanModel initialisation.
    """

    def __init__(self, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to CointegrationModel.")
        self.cfg = cfg

    def fit(self, df_window: pd.DataFrame) -> dict:
        results = {}
        
        # ADF tests (Level & Diff)
        for col in df_window.columns:
            logging.debug(f"Trend for {col}: {self.cfg.adf_trend}")
            lvl = ADF(df_window[col], trend=self.cfg.adf_trend, method="bic")
            logging.debug(f"lvl: {lvl}")
            df1 = ADF(df_window[col].diff().dropna(), trend=self.cfg.adf_trend, method="bic")
            logging.debug(f"df1: {df1}")
            results[f"ADF_{col}_lvl_stat"] = lvl.stat
            results[f"ADF_{col}_lvl_crit"] = lvl.critical_values['5%']
            results[f"ADF_{col}_dif_stat"] = df1.stat
            results[f"ADF_{col}_dif_crit"] = df1.critical_values['5%']

            if lvl.stat < lvl.critical_values['5%']:
                logging.debug(f"if lvl")
                if self.cfg.coint_enforce_i1:
                    raise DiagnosticFailure(
                        stage="i1_check",
                        reason=f"'{col}' appears I(0) in this window (ADF stat {lvl.stat:.3f} < crit {lvl.critical_values['5%']:.3f}).",
                        stats={"column": col, "adf_stat": lvl.stat, "adf_crit_5pct": lvl.critical_values['5%']},
                        series={col: df_window[col]},
                    )
                    # raise ValueError(
                    #     f"'{col}' appears I(0) in this window "
                    #     f"(ADF stat {lvl.stat:.3f} < crit {lvl.critical_values['5%']:.3f}). "
                    #     f"VECM requires I(1) inputs."
                    # )
                else:
                    logging.warning(
                        f"'{col}' appears I(0) in this window, continuing because "
                        f"coint_enforce_i1=False."
                    )
            if df1.stat > df1.critical_values['5%']:        # fails to reject after differencing -> I(2)+
                logging.warning(
                    f"'{col}' may be I(2) in this window (ADF on diff: "
                    f"{df1.stat:.3f} > crit {df1.critical_values['5%']:.3f}). Proceeding with caution."
                )

        # Phillips Ouliaris test
        po = phillips_ouliaris(
                            df_window[self.cfg.dep_col], df_window[self.cfg.indep_cols], trend=self.cfg.adf_trend, test_type="Za", kernel="bartlett"
                        )
        results["PO_stat"] = po.stat
        results["PO_crit"] = po.critical_values[5]
        results["PO_p_value"] = po.pvalue

        if po.stat > po.critical_values[5]:
            if self.cfg.coint_enforce_po:
                raise DiagnosticFailure(
                    stage="phillips_ouliaris",
                    reason=f"PO t-stat {po.stat:.3f} > crit {po.critical_values[5]:.3f}. Pair may not be cointegrated.",
                    stats={"po_stat": po.stat, "po_crit_5pct": po.critical_values[5], "po_pvalue": po.pvalue},
                    series={self.cfg.dep_col: df_window[self.cfg.dep_col],
                            **{c: df_window[c] for c in self.cfg.indep_cols}},
                )
                # raise ValueError(
                #     f"PO t-stat {po.stat:.3f} > crit {po.critical_values[5]:.3f}). "
                #     f"Pairs may not be co-integrated."
                # )
            else:
                logging.debug(
                    f"PO test failed (stat {po.stat:.3f} > crit "
                    f"{po.critical_values[5]:.3f}) — continuing because "
                    f"coint_enforce_po=False."
                )

        # VAR selection
        k_ar_diff = VAR(df_window).select_order(self.cfg.coint_maxlags).bic
        results["VAR_lag_bic"] = k_ar_diff
        logging.debug(f"VAR selection: {k_ar_diff}")

        # Johansen test
        jtest = coint_johansen(df_window, det_order=self.cfg.johansen_det, k_ar_diff=k_ar_diff)
        results["johansen"] = jtest
        logging.debug(f"Johansen test: {jtest}")

        # Dynamically determine the number of ranks to test
        num_cols = len(df_window.columns)
        results["Johansen_r=0_stat"] = jtest.trace_stat[0]
        results[f"Johansen_r=0_crit_95"] = jtest.trace_stat_crit_vals[0, 1]
        for i in range(1, num_cols):
            results[f"Johansen_r<={i}_stat"] = jtest.trace_stat[i]
            results[f"Johansen_r<={i}_crit_95"] = jtest.trace_stat_crit_vals[i, 1]

        if results["Johansen_r=0_stat"] <= results["Johansen_r=0_crit_95"]:
            if self.cfg.coint_enforce_johansen:
                raise DiagnosticFailure(
                    stage="johansen",
                    reason=(f"Johansen trace fails to reject r=0 (stat={results['Johansen_r=0_stat']:.3f} "
                            f"<= crit={results['Johansen_r=0_crit_95']:.3f})."),
                    stats={"trace_stat": results["Johansen_r=0_stat"], "trace_crit_95": results["Johansen_r=0_crit_95"]},
                    series={self.cfg.dep_col: df_window[self.cfg.dep_col],
                            **{c: df_window[c] for c in self.cfg.indep_cols}},
                )
                # raise ValueError(
                #     f"Johansen trace test fails to reject r=0 "
                #     f"(stat={results['Johansen_r=0_stat']:.3f} <= "
                #     f"crit={results['Johansen_r=0_crit_95']:.3f}). "
                #     f"No evidence of cointegration at the 95% level."
                # )
            else:
                logging.warning(
                    f"Johansen trace test fails to reject r=0 — proceeding "
                    f"because coint_enforce_johansen=False."
                )

        # VECM coefficients
        vecm_res = VECM(df_window, k_ar_diff=k_ar_diff, deterministic=self.cfg.vecm_det).fit()
        results["vecm"] = vecm_res
        raw_beta = vecm_res.beta[:, 0]
        if self.cfg.vecm_det == "n":
            raw_alpha = 0.0
        else:
            raw_alpha = float(vecm_res.const_coint.flat[0])
        results["VECM_beta"] = raw_beta # Cointegrating vector: beta is normalised so first variable has coeff 1.
        results["VECM_alpha"] = raw_alpha

        # Residual (Spread) stationarity
        spread = df_window.values @ raw_beta + raw_alpha
        logging.debug(f"Spread stationarity: {spread}")

        #Half-Life
        # Spread_t = rho * Spread_{t-1} + e
        # Half-life = -log(2) / log(abs(rho))
        spread_series = pd.Series(spread)
        z_lag = spread_series.shift(1).dropna()
        z_diff = spread_series.diff().dropna()
        # Simple OLS: delta_z = alpha + beta * z_{t-1}
        # Discrete version of the OU process
        if self.cfg.ols_spread_const:
            reg = sm.OLS(z_diff, sm.add_constant(z_lag)).fit()
            lambda_val = reg.params.iloc[1]
        else:
            reg = sm.OLS(z_diff, z_lag).fit()
            lambda_val = reg.params.iloc[0]
        if lambda_val < 0:
            results["half_life"] = -np.log(2) / lambda_val
        else:
            results["half_life"] = 1000.0  # Cap it if it's not reverting

        spread_adf = ADF(spread, trend="n", method="bic") # Trend is none after VECM fitting
        results["Spread_ADF_stat"] = spread_adf.stat
        results["Spread_ADF_crit_95"] = spread_adf.critical_values['5%']
        results["Spread_pvalue"] = spread_adf.pvalue

        logging.debug(f"Spread ADF: {spread_adf}")
        return results
        
    def get_model_confidence(self, results: dict) -> dict:
        """
        Normalises test results into a 0-1 range.
        Values > 1.0 indicate the test passed.
        Values < 1.0 indicate how close it was to passing.
        """
        conf = {}
        
        # Johansen strength (ratio of stat to 95% crit val)
        # If > 1, the null of 'no cointegration' is rejected.
        conf['johansen_strength'] = results["Johansen_r=0_stat"] / results["Johansen_r=0_crit_95"]
        
        # Residual stationarity strength
        # We use (1 - p-value) so that 1.0 is perfectly stationary and 0.0 is a unit root.
        conf['resid_stationarity'] = 1 - results["Spread_pvalue"]
        
        # Distance from critical value (ADF)
        # How many times larger is our test stat than the critical value?
        conf['resid_adf_ratio'] = abs(results["Spread_ADF_stat"]) / abs(results["Spread_ADF_crit_95"])

        # Total probability score
        weights = {'johansen_strength': 0.4, 'resid_stationarity': 0.6}
        
        # Cap strengths at 1.2 to avoid outliers skewing the ML
        norm_j = min(conf['johansen_strength'], 1.2)
        norm_s = min(conf['resid_stationarity'], 1.0)
        
        conf['total_confidence'] = (norm_j * weights['johansen_strength']) + (norm_s * weights['resid_stationarity'])
        
        return conf
    
    def check_health(self, df_window: pd.DataFrame, beta: np.ndarray, alpha: float) -> float:
        """
        Returns the p-value of the ADF test on the spread 
        calculated with CURRENT model parameters.
        """
        # Calculate the spread using current Kalman weights: spread = Y - (X * beta)
        # If your beta includes the intercept, ensure X has the constant.
        # Otherwise: spread = Y - (X @ beta_cols + alpha)
        
        # Assuming df_window columns match your model order
        y = df_window[self.cfg.dep_col].values
        X = df_window[self.cfg.indep_cols].values
        
        # Calculate spread: Y_t - (beta1*X1 + beta2*X2 + ... + alpha)
        spread = y - (X @ beta + alpha)
        
        # Run ADF on the residual. We keep trend="n" to detect regime break
        try:
            res = ADF(spread, trend="n", method="bic")
            return res.pvalue
        except:
            return 1.0 # Return failure if test crashes
  