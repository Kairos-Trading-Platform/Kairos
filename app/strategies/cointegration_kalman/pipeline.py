import logging # TODO logger
from pathlib import Path
import pandas as pd
import numpy as np
import dataclasses
from itertools import combinations
import os
from sklearn.model_selection import train_test_split
import yfinance as yf
from .data_handler import DataHandler
from .config import Config
from .cointegration import CointegrationModel
from .kalman import KalmanModel
from .portfolio import Portfolio
from .signal_engine import SignalEngine
from .koopman import KoopmanRegimeFilter
from .quality_models import RandomForestModel, XGBoostModel
from .executor import StrategyExecutor

def run_production(data: DataHandler,
                   cfg: Config,
                   replay: bool = False) -> pd.DataFrame:
    """
    Steps the Kalman filter forward over all unseen dates.
    replay=True  : runs without saving state (safe to re-run anytime).
    replay=False : advances kf.last_date and persists state after each step.
    Returns a DataFrame with the same schema as Backtester.run().
    Uses the DataHandler's combined data to ensure ADF windows have sufficient history.
    """
    # ------------------------------------------------------------------
    # Initialise all components from persisted state
    # ------------------------------------------------------------------
    try:
        # Load existing state and parameters
        kf = KalmanModel.load_state(cfg=cfg)
    except Exception as e:
        logging.error(f"Could not load Kalman state: {e}")
        return pd.DataFrame()

    logging.info(f"Kalman state loaded from {cfg.state_file} "
                 f"(last_date={kf.last_date})")

    port = Portfolio(cfg=cfg)
    coint_model = CointegrationModel(cfg=cfg)
    se = SignalEngine(entry_z=kf.entry_z, cfg=cfg)
    koopman      = KoopmanRegimeFilter(cfg=cfg)

    # Load fitted ML models. gate_signal is a no-op if is_fitted=False,
    # so passing None is safe, but loading from disk is preferred.
    qm = None
    if Path(cfg.qm_file).exists():
        try:
            qm = RandomForestModel.load(cfg=cfg)
            logging.info(f"RF model loaded (threshold={qm.threshold:.2f}, "
                         f"fitted={qm.is_fitted})")
        except Exception as e:
            logging.warning(f"Could not load RF model: {e}")

    xgb_model = None
    if Path(cfg.xgb_file).exists():
        try:
            xgb_model = XGBoostModel.load(cfg=cfg)
            logging.info(f"XGB model loaded (threshold={xgb_model.threshold:.2f}, "
                         f"fitted={xgb_model.is_fitted})")
        except Exception as e:
            logging.warning(f"Could not load XGB model: {e}")

    executor = StrategyExecutor(
        kf=kf,
        signal_engine=se,
        portfolio=port,
        quality_model=qm,
        xgb_model=xgb_model,
        koopman_filter=koopman,
        cfg=cfg,
    )

    # ------------------------------------------------------------------
    # Determine which bars to process
    # ------------------------------------------------------------------

    if replay:
        # Re-run exactly the test period, start where training ended
        start_idx = data.train_test_boundary
        to_process = data.df.iloc[start_idx:]
    else:
        if kf.last_date is None:
            # If no state, start at the earliest possible window
            start_idx = cfg.bt_window
            to_process = data.df.iloc[start_idx:]
        else:
            # Start from the day after the last recorded state
            to_process = data.df[data.df.index > kf.last_date]
            if to_process.empty:
                logging.info("No new data to process.")
                return pd.DataFrame()
    
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    records = []
    z_history = list(kf._z_score_history)
    for current_date, _ in to_process.iterrows():
        # locate the integer position so get_observation can slice by idx
        idx = data.df.index.get_loc(current_date)
        df_window = data.get_window(idx, cfg.bt_window)
        if len(df_window) < 5: 
            logging.error(f"Skipping {current_date}: Window size {len(df_window)} too small for ADF.")
            continue
        y_t, X_t = data.get_observation(cfg.dep_col, cfg.indep_cols, idx=idx)

        # Health check: Kalman spread vs VECM-theoretical spread stationarity
        p_traded = coint_model.check_health(df_window, kf.m[:-1], kf.m[-1])

        # Feature extraction for ML gates
        features_ok = False
        try:
            c_res = coint_model.fit(df_window)
            p_theoretical = c_res["Spread_pvalue"]
            logging.info(f"Date: {current_date} | Traded P: {p_traded:.4f} | Theoretical P: {p_theoretical:.4f} | Drift: {abs(p_traded - p_theoretical):.4f}")
            ext_feat = {
                "feat_johansen_trace": c_res["johansen"].trace_stat[0],
                cfg.HALF_LIFE: c_res.get("half_life", 20.0)
            }
            features_ok = True
        except Exception as e:
            logging.warning(
                f"{current_date}: feature extraction failed ({e}). "
                f"ML gates will receive zero features — entries blocked by convention."
            )
            ext_feat = {
                cfg.JOHANSEN_TRACE: 0.0,
                cfg.HALF_LIFE: 0.0
            }

        # P&L uses next-bar returns
        if idx + 1 >= len(data.df):
            continue
        log_ret    = data.df.iloc[idx + 1].values - data.df.iloc[idx].values
        raw_prices = np.exp(data.df.iloc[idx + 1][[cfg.dep_col] + cfg.indep_cols].values)
        #log_ret     = data.df.iloc[idx].values - data.df.iloc[idx - 1].values
        prev_signal = records[-1][cfg.SIGNAL] if records else 0
        #raw_prices  = np.exp(data.df.iloc[idx][[dep_col] + indep_cols].values)

        # Generate signal
        step_record = executor.execute_step(y_t, X_t, log_ret, raw_prices, prev_signal, external_features=ext_feat)
        step_record[cfg.datetime_col] = current_date
        records.append(step_record)

        # Append z_t
        z_history.append(step_record[cfg.Z_SCORE])

        # Update entry for the next bar
        if len(z_history) >= cfg.kalman_norm_window:
            se.entry_z = float(np.percentile(
                np.abs(z_history[-cfg.kalman_norm_window:]),
                cfg.entry_z_percentile
            ))

        if not replay:
            kf.last_date = current_date
            kf.save_state(cfg.state_file)

        current_sig = step_record[cfg.SIGNAL]
        # Determine the conviction/size label
        conviction = abs(current_sig)
        if conviction == 1.0:
            size_label = "(FULL)"
        elif conviction > 0:
            size_label = f"(SCALED: {conviction:.1%})"
        else:
            size_label = ""

        # Determine the directional label
        if current_sig > 0:
            direction = f"LONG {cfg.dep_col} / SHORT {cfg.indep_cols[0]}"
        elif current_sig < 0:
            direction = f"SHORT {cfg.dep_col} / LONG {cfg.indep_cols[0]}"
        else:
            direction = "FLAT"

        # Print with the new dynamic action string
        if direction != "FLAT":
            logging.info(f"\n Date: {current_date} | {direction} {size_label} | Shares dep: {float(step_record[cfg.SHARES_DEP]):.2f} "
                f"| Shares indep: {step_record[cfg.SHARES_INDEP]} Notional: ${step_record[cfg.NOTIONAL]:,.2f} | features_ok={features_ok}")
    
    if not records:
        logging.info("No records generated.")
        return pd.DataFrame()
    return pd.DataFrame(records).set_index(cfg.datetime_col)

def compute_and_save_all_pairs(df_all, cfg = Config):
    """
    Compute and save co-integration analysis for all unique pairs of columns in df_all.
    """
    columns = df_all.columns.tolist()
    pairs = list(combinations(columns, 2))

    for dep_col, indep_col in pairs:
        pair_cfg = dataclasses.replace(
                cfg, dep_col=dep_col, indep_cols=[indep_col]
            )
        logging.debug(f"Pair processed: {dep_col} vs {indep_col}")
        summary_filename = os.path.join(pair_cfg.output_dir, f"summary_{dep_col}_vs_{indep_col}.csv")
        state_filename = os.path.join(pair_cfg.output_dir, f"model_state_{dep_col}_vs_{indep_col}.pkl")

        if os.path.exists(summary_filename) and os.path.exists(state_filename):
                logging.info(f"Results already exist for {dep_col} vs {indep_col}. Skipping.")
                continue

        logging.info(f"\nProcessing pair: {dep_col} vs {indep_col}")
        
        selected_columns = [dep_col, indep_col]

        # Split data
        df_raw = df_all[selected_columns]
        df_train, df_test = train_test_split(df_raw, test_size=0.3, random_state=42, shuffle=False)
        #logging.info("Data split")

        # Prepare data
        data = DataHandler(cfg=pair_cfg, df=df_train, log=pair_cfg.log_prices)
        data.summary()

        # Fit model
        model = CointegrationModel(cfg=pair_cfg)
        res = model.fit(data.df)
        logging.info("Model fit")

        confidence_metrics = model.get_model_confidence(res)
        res.update({f"CONF_{k}": v for k, v in confidence_metrics.items()})
        summary_df = pd.DataFrame.from_dict(res, orient='index', columns=['Value'])
        logging.info("Data summary created")

        # Save summary
        summary_df.to_csv(summary_filename)
        logging.info(f"Summary saved to: {summary_filename}")
    else:
        logging.info(f"Results already exist for {dep_col} vs {indep_col}. Skipping.")

def download():
    
    tickers_list=["^GSPC", "NQ=F", "EURUSD=X", "JPYUSD=X", "CHF=X", "AUDUSD=X", "NZDUSD=X"]
    prices = yf.download(tickers_list, start='2007-01-01', end='2026-04-15', group_by='ticker', auto_adjust=True, progress=False)
    if not prices.empty:
        try:
            close_prices = prices.xs('Close', axis=1, level=1)
            
            # 3. Save to CSV
            close_prices.to_csv('data.csv', index=True)
            logging.info("Success! Data saved to data_gwp.csv")
            logging.debug(close_prices.tail())
        except KeyError:
            logging.error("Error: Could not find 'Close' columns in the downloaded data.")
    else:
        logging.warning("No data was downloaded.")
    exit()

def find_first_significant_window(df_log_prices: pd.DataFrame,
                                   cfg: Config,
                                   step_size: int = 5,
                                   confirm_n: int = 2,
                                   confirm_frac: float = 0.60) -> dict | None:
    """
    Scans window lengths from coint_min_window and returns the first
    window that achieves Spread_pvalue < 0.05 AND is stable across
    confirm_n neighbouring windows on each side.

    The z_score_min_bars constraint does NOT apply here — that is
    enforced downstream in automate_strategy_params. This function
    is only responsible for finding cointegration.
    """
    total_len    = len(df_log_prices)
    # Minimum: enough rows for VECM with maxlags, not z_score_min_bars
    min_window   = max(getattr(cfg, 'coint_min_window', 60),
                       cfg.coint_maxlags * 10)
    coint_tester = CointegrationModel(cfg=cfg)

    logging.info(f"\n--- Searching for First Significant Window (p < 0.05) ---")
    logging.info(f"min_window={min_window} | total={total_len} | "
                 f"step={step_size} | confirm_n={confirm_n} | "
                 f"confirm_frac={confirm_frac}")

    def _test_window(w: int) -> float | None:
        if w < min_window or w > total_len:
            return None
        try:
            res = coint_tester.fit(df_log_prices.tail(w))
            return res.get("Spread_pvalue", 1.0)
        except Exception as e:
            logging.debug(f"Window {w} failed: {e}")
            return None

    for w in range(min_window, total_len + 1, step_size):
        p_val = _test_window(w)

        if p_val is None or p_val >= 0.05:
            if w % 100 == 0:
                logging.info(f"Window {w}: p={f'{p_val:.4f}' if p_val is not None else 'N/A'}")
            continue

        # Candidate — confirm stability across neighbours
        neighbours = [w + k * step_size
                      for k in range(-confirm_n, confirm_n + 1)
                      if k != 0]
        neighbour_results = []
        for nb in neighbours:
            nb_p = _test_window(nb)
            neighbour_results.append((nb, nb_p, nb_p is not None and nb_p < 0.05))

        n_pass   = sum(r[2] for r in neighbour_results)
        n_tested = sum(1 for r in neighbour_results if r[1] is not None)
        if n_tested < confirm_n:
            logging.info(
                f"Window {w}: only {n_tested} neighbours testable "
                f"(need {confirm_n}). Continuing search."
            )
            continue

        frac_pass = n_pass / n_tested if n_tested > 0 else 0.0

        logging.info(
            f"Candidate window {w}: p={p_val:.4f} | "
            f"neighbours passing: {n_pass}/{n_tested} ({frac_pass:.0%})"
        )

        if frac_pass >= confirm_frac:
            # Take the window with lowest p-value among all passing windows
            all_passing = [(w, p_val)] + [
                (nb, nb_p)
                for nb, nb_p, passed in neighbour_results
                if passed and nb_p is not None
            ]
            best_w, best_p = min(all_passing, key=lambda x: x[1])

            try:
                best_res = coint_tester.fit(df_log_prices.tail(best_w))
            except Exception:
                best_res = coint_tester.fit(df_log_prices.tail(w))
                best_w, best_p = w, p_val

            logging.info(
                f"Confirmed! Best window: {best_w} | p={best_p:.4f} | "
                f"half-life={best_res.get('half_life', float('nan')):.2f}"
            )
            return {
                'window_length': best_w,
                'p_value':       best_p,
                'half_life':     best_res.get('half_life'),
                'johansen_stat': best_res.get('Johansen_r=0_stat'),
                'confirm_frac':  frac_pass,
                'n_neighbours':  n_tested,
            }
        else:
            logging.info(
                f"Window {w} rejected — only {frac_pass:.0%} of neighbours "
                f"pass (need {confirm_frac:.0%}). Continuing search."
            )

    logging.error("No stable cointegration window found.")
    return None

def automate_strategy_params(cfg: Config,
                              match_results: dict) -> bool:
    """
    Sets Kalman and signal parameters from the detected cointegration
    window. Returns True on success, False if the window is too short
    to satisfy z_score constraints even after tolerance relaxation.
    """
    window    = match_results['window_length']
    half_life = match_results['half_life']

    # z_score_min_bars: recompute from current tol/conf settings
    # (may have changed since Config was constructed)
    cfg.z_score_min_bars = cfg._get_z_score_min_bar()

    # Max burn-in: 25% of window; max norm_window: 50% of window
    max_burn_in    = window // 4
    max_norm_window = window // 2

    # Relax tolerance if z_score_min_bars exceeds max_burn_in
    original_tol = cfg.z_score_std_tol
    for candidate_tol in np.arange(original_tol, 0.51, 0.01):
        cfg.z_score_std_tol  = round(float(candidate_tol), 2)
        cfg.z_score_min_bars = cfg._get_z_score_min_bar()
        if cfg.z_score_min_bars <= max_burn_in:
            break
    else:
        logging.error(
            f"window_length={window} is too short to satisfy z_score "
            f"constraints even at tol=0.50. "
            f"Minimum viable window ≈ {cfg.z_score_min_bars * 4} bars. "
            f"The search found a cointegration window but it is too short "
            f"to trade reliably. Consider using a longer price history."
        )
        return False

    if cfg.z_score_std_tol > original_tol:
        logging.warning(
            f"z_score_std_tol relaxed from {original_tol:.2f} to "
            f"{cfg.z_score_std_tol:.2f} to fit window_length={window}."
        )

    # Set parameters — all constraints now satisfiable
    cfg.kalman_norm_window = max(cfg.z_score_min_bars,
                                 min(max_norm_window, window))
    cfg.kalman_burn_in     = cfg.z_score_min_bars
    cfg.coint_maxlags      = max(5, int(np.power(window, 1/3)))
    cfg.regime_window      = max(10, int(half_life * 0.5))

    logging.info(
        f"--- Parameters Automated ---\n"
        f"  Window: {window} | Norm Window: {cfg.kalman_norm_window} | "
        f"Burn-in: {cfg.kalman_burn_in}\n"
        f"  z_score_min_bars: {cfg.z_score_min_bars} | "
        f"tol: {cfg.z_score_std_tol:.2f}\n"
        f"  Max Lags: {cfg.coint_maxlags} | "
        f"Regime Window: {cfg.regime_window}"
    )
    return True
