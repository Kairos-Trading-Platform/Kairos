import logging # TODO logger
import numpy as np
import pandas as pd
from statsmodels.tools import add_constant
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from .data_handler import DataHandler
from .config import Config
from .performance import Performance
from .cointegration import CointegrationModel
from .kalman import KalmanModel,KalmanMLE
from .backtester import Backtester
from .quality_models import RandomForestModel, XGBoostModel

class WalkForwardValidator:
    """
    Expanding-window walk-forward validation.

    Splits the training set into `n_splits` folds where the training
    window grows with each fold (no data leakage).  For each fold it:
      1. Re-fits CointegrationModel on the current training slice.
      2. Re-runs KalmanMLE to get fresh noise hyperparameters.
      3. Runs Backtester and collects performance metrics.
      4. Trains RF + XGBoost classifiers on that fold's labels.

    At the end it aggregates fold metrics and returns a summary DataFrame,
    giving you an honest estimate of out-of-sample generalisation before
    you touch the held-out test set.

    Parameters (in Config)
    ----------------------
    wf_n_splits       : int   number of folds (default 5)
    wf_min_train_frac : float minimum fraction of data used in fold 1 (default 0.40)
    """

    def __init__(self, data: DataHandler, cfg: Config,
                 beta_init: np.ndarray, alpha_init: float,
                 opt_W_beta: float, opt_W_alpha: float, opt_V: float,
                 opt_nu: float = None):
        self.data       = data
        self.cfg        = cfg
        self.beta_init  = beta_init
        self.alpha_init = alpha_init
        self.opt_W_beta  = opt_W_beta
        self.opt_W_alpha = opt_W_alpha
        self.opt_V       = opt_V
        self.opt_nu      = opt_nu
        self.fold_metrics: list[dict] = []

    def run(self) -> pd.DataFrame:
        """
        Execute all folds.  Returns a DataFrame with one row per fold
        showing Sharpe, MaxDrawdown, WinRate, NumTrades, and classifier
        CV f1 scores.
        """
        n     = len(self.data.df)
        cfg   = self.cfg
        perf  = Performance(cfg=cfg)

        min_train = max(int(n * cfg.wf_min_train_frac), cfg.bt_window)
        # Expanding fold boundaries: train ends grow from min_train to n
        fold_ends = np.linspace(min_train, n, cfg.wf_n_splits + 1, dtype=int)[1:]

        logging.info(f"\n{'='*60}")
        logging.info(f"WALK-FORWARD VALIDATION  ({cfg.wf_n_splits} folds)")
        logging.info(f"{'='*60}")

        for fold_idx, train_end in enumerate(fold_ends):
            logging.info(f"\n--- Fold {fold_idx + 1}/{cfg.wf_n_splits} "
                         f"| train rows: {train_end} ---")

            # Slice training data for this fold
            fold_df   = self.data.df.iloc[:train_end]
            fold_data = DataHandler.__new__(DataHandler)
            fold_data.log = self.data.log
            fold_data.df  = fold_df

            # Re-fit cointegration on this fold's window
            coint  = CointegrationModel(cfg=cfg)
            try:
                res    = coint.fit(fold_df.tail(cfg.bt_window))
                beta   = res["VECM_beta"][1:]
                alpha  = res["VECM_alpha"]
            except Exception as exc:
                logging.debug(f"  Fold {fold_idx+1}: coint failed ({exc}), using prior beta/alpha.")
                beta, alpha = self.beta_init, self.alpha_init

            # Re-run MLE on this fold (optional but gives fold-specific params)
            y_fold = fold_df[cfg.dep_col].values
            X_fold = add_constant(fold_df[cfg.indep_cols], prepend=False).values
            try:
                mle    = KalmanMLE(init_beta=beta, init_alpha=alpha, cfg=cfg)
                opt    = mle.fit(y_fold, X_fold)
                w_beta, w_alpha, v = opt[0], opt[cfg.state_dim - 1], opt[cfg.state_dim]
                nu     = opt[cfg.state_dim + 1] if cfg.dist_shape == 'student' else None
            except Exception as exc:
                logging.warning(f"  Fold {fold_idx+1}: MLE failed ({exc}), using global params.")
                w_beta, w_alpha, v, nu = self.opt_W_beta, self.opt_W_alpha, self.opt_V, self.opt_nu

            # Build fresh Kalman filter for this fold
            kf_fold = KalmanModel(
                W_diag=[w_beta] * (cfg.state_dim - 1) + [w_alpha],
                V=v, cfg=cfg
            )
            kf_fold.nu = nu

            # Derive entry_z from a short Kalman run on the fold data
            kf_probe = KalmanModel(
                W_diag=[w_beta] * (cfg.state_dim - 1) + [w_alpha],
                V=v, cfg=cfg
            )
            kf_probe.initialise(beta, alpha)
            df_hist = kf_probe.run_and_report(
                y_fold, X_fold,
                burn_in=cfg.kalman_burn_in, nu=nu,
                dates=fold_df.index
            )
            z_train = df_hist["z_score"].dropna()
            entry_z = np.percentile(np.abs(z_train), cfg.entry_z_percentile)

            # Backtest on this fold
            bt = Backtester(data=fold_data, kf=kf_fold, cfg=cfg, entry_z=entry_z)
            bt_res = None
            try:
                bt_res    = bt.run()
                fold_perf = perf.evaluate(bt_res, label=f"wf_fold_{fold_idx+1}")
            except Exception as exc:
                logging.warning(
                    f"  Fold {fold_idx+1}: backtest or evaluation failed ({exc}). "
                    f"Recording NaN sentinel row."
                )
                fold_perf = {
                    "Sharpe": np.nan, "Calmar": np.nan, "MaxDrawdown": np.nan,
                    "TotalReturn": np.nan, "AnnReturn": np.nan, "Turnover": np.nan,
                    "WinRate": np.nan, "NumTrades": 0,
                }

            rf_f1, xgb_f1 = np.nan, np.nan
            if bt_res is not None:
                # Train RF and XGB classifiers on this fold's labels
                labels   = bt.build_labels()
                print(f"Label value counts: {labels.value_counts(dropna=False).to_dict()}")
                features = bt.get_feature_matrix()
                print("\nFeature matrix stats:")
                print(features.describe().round(6))
                print("\nNaN counts:", features.isna().sum().to_dict())
                print("Constant cols:", [c for c in features.columns if features[c].nunique() <= 1])

                valid_labels = labels.dropna()
                if (len(valid_labels) >= 6 
                        and len(np.unique(valid_labels)) == 2
                        and np.bincount(valid_labels.astype(int).values).min() >= 2):
                    # RF
                    rf = RandomForestModel(cfg=cfg)
                    rf.train(features, labels)
                    common_idx = features.index.intersection(labels.dropna().index)
                    X_full = features.loc[common_idx].fillna(0).values
                    y_full = labels.loc[common_idx].fillna(0).astype(int).values
                    rf_preds  = rf.model.predict(X_full)
                    rf_rep    = classification_report(y_full, rf_preds, output_dict=True, zero_division=0)
                    rf_f1 = rf_rep.get('1', {}).get('f1-score', np.nan)

                    # XGB
                    xgb = XGBoostModel(cfg=cfg)
                    xgb.train(features, labels)
                    try:
                        xgb_proba = xgb.model.predict_proba(X_full)[:, 1]
                        xgb_preds = (xgb_proba >= 0.5).astype(int)
                    except AttributeError:
                        xgb_preds = (xgb.model.predict(X_full) > 0.5).astype(int)
                    xgb_rep   = classification_report(
                        y_full, xgb_preds,
                        output_dict=True, zero_division=0
                    )
                    xgb_f1 = xgb_rep.get('1', {}).get('f1-score', np.nan)

            self.fold_metrics.append({
                "fold":        fold_idx + 1,
                "train_rows":  train_end,
                "entry_z":     entry_z,
                **fold_perf,
                "rf_f1":       rf_f1,
                "xgb_f1":      xgb_f1,
            })

        summary = pd.DataFrame(self.fold_metrics)
        self._print_summary(summary)
        self._plot_wf_equity(summary)
        return summary

    def _print_summary(self, summary: pd.DataFrame):
        logging.info(f"\n{'='*60}")
        logging.info("WALK-FORWARD SUMMARY")
        logging.info(f"{'='*60}")
        cols = ["fold", "Sharpe", "MaxDrawdown", "WinRate", "NumTrades", "rf_f1", "xgb_f1"]
        available = [c for c in cols if c in summary.columns]
        logging.info(f"\n{summary[available].round(4).to_string(index=False)}")
        logging.info(f"\nMean Sharpe:  {summary['Sharpe'].mean():.4f}  "
                     f"± {summary['Sharpe'].std():.4f}")
        logging.info(f"Mean MaxDD:   {summary['MaxDrawdown'].mean():.4f}")
        if 'rf_f1' in summary:
            logging.info(f"Mean RF F1:   {summary['rf_f1'].mean():.4f}")
        if 'xgb_f1' in summary:
            logging.info(f"Mean XGB F1:  {summary['xgb_f1'].mean():.4f}")

    def _plot_wf_equity(self, summary: pd.DataFrame):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].bar(summary["fold"], summary["Sharpe"], color="steelblue")
        axes[0].axhline(summary["Sharpe"].mean(), color="red", linestyle="--",
                        label=f"Mean={summary['Sharpe'].mean():.2f}")
        axes[0].set_title("Walk-Forward Sharpe by Fold")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "rf_f1" in summary.columns and "xgb_f1" in summary.columns:
            axes[1].plot(summary["fold"], summary["rf_f1"],  "o-", label="RF F1")
            axes[1].plot(summary["fold"], summary["xgb_f1"], "s-", label="XGB F1")
            axes[1].set_title("Classifier F1 (class=1 / profitable trades) by Fold")
            axes[1].set_ylabel("F1 Score")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fpath = os.path.join(self.cfg.output_dir, "walk_forward_summary.png")
        plt.savefig(fpath, dpi=150)
        plt.close()
        logging.info(f"Walk-forward plot saved to {fpath}")
