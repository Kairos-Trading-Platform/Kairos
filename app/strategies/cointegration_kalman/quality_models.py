"""
ML-based signal quality gates.

RandomForestModel and XGBoostModel share the same
interface: train on Backtester output, then gate_signal() scales/blocks a
raw ±1 signal based on predicted trade quality. 
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import joblib
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


class QualityModelBase(ABC):
    """
    Common interface for ML-based signal quality gates.

    Subclasses must set `threshold_attr` / `file_attr` (names of the
    corresponding fields on Config) and implement `train()`. Everything
    else — gate_signal, save, load, feature-vector handling — is shared.
    """

    threshold_attr: str = None   # e.g. "rf_threshold" / "xgb_threshold"
    file_attr: str = None        # e.g. "qm_file" / "xgb_file"

    def __init__(self, cfg=None, n_splits: int = None):
        if cfg is None:
            raise ValueError(
                f"A valid Config instance must be provided to {type(self).__name__}."
            )
        self.cfg = cfg
        self.model = None
        self.threshold = getattr(cfg, self.threshold_attr)
        self.n_splits = n_splits or cfg.rf_n_splits
        self.is_fitted = False
        self.feature_cols = None
        self.feat_template = None

    @abstractmethod
    def train(self, features: pd.DataFrame, labels: pd.Series):
        ...

    def _feature_vector(self, feature_row) -> np.ndarray:
        if isinstance(feature_row, pd.Series):
            return feature_row.reindex(self.feature_cols).fillna(0).values
        return np.nan_to_num(np.array(feature_row, dtype=float))

    def gate_signal(self, signal, feature_row):
        """Returns a scaled signal based on model confidence.
        0.0 = Skip, 0.5 = Half-size, 1.0 = Full-size."""
        if signal == 0 or not self.is_fitted:
            return signal
        x = self._feature_vector(feature_row)
        prob = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        if prob >= 0.70:
            return float(signal)
        elif prob >= self.threshold:
            return signal * 0.5
        else:
            return 0.0

    def save(self, filepath: str = None):
        filepath = filepath or getattr(self.cfg, self.file_attr)
        joblib.dump({
            "model": self.model,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted,
            "feature_cols": self.feature_cols,
            "feat_template": self.feat_template,
        }, filepath)

    @classmethod
    def load(cls, cfg) -> "QualityModelBase":
        data = joblib.load(getattr(cfg, cls.file_attr))
        instance = cls(cfg=cfg)
        instance.model = data["model"]
        instance.threshold = data["threshold"]
        instance.is_fitted = data["is_fitted"]
        instance.feature_cols = data["feature_cols"]
        instance.feat_template = data["feat_template"]
        return instance


class RandomForestModel(QualityModelBase):
    """
    Random-forest classifier trained on Backtester output to gate
    low-quality signal entries.

    Full loop:
        Backtester.run()          -> results  (features + PnL)
        Backtester.build_labels() -> labels   (1=profitable, 0=not)
        RandomForestModel.train()            -> fitted RF
        (live) gate_signal()                  -> filtered position
    """

    threshold_attr = "rf_threshold"
    file_attr = "qm_file"

    def __init__(self, cfg=None, n_estimators: int = None, n_splits: int = None):
        super().__init__(cfg=cfg, n_splits=n_splits)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators or cfg.rf_n_estimators,
            max_features="sqrt",
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        )

    def train(self, features: pd.DataFrame, labels: pd.Series):
        """
        Trains using time-series cross-validation so future data never
        leaks into training folds. Only active-signal rows are used.
        """
        self.feature_cols = list(features.columns)
        self.feat_template = pd.Series(np.nan, index=self.feature_cols)
        mask = labels.notna()
        X = features.loc[mask].fillna(0).values
        y = labels.loc[mask].values.astype(int)

        if len(np.unique(y)) < 2:
            logging.warning("Warning: only one class in labels. RF not trained.")
            return

        logging.info(f"\nClass distribution: {np.bincount(y)}")
        logging.info(f"Positive rate: {y.mean():.3f}")

        n_splits = 2 if y.sum() < self.n_splits * 2 else self.n_splits
        logging.warning(
            f"Warning: only {y.sum()} positive samples for "
            f"{self.n_splits}-fold CV. Reducing to 2 folds."
        )
        tscv = TimeSeriesSplit(n_splits=n_splits)

        logging.info("Walk-forward CV:")
        for fold, (tr, va) in enumerate(tscv.split(X)):
            self.model.fit(X[tr], y[tr])
            preds = self.model.predict(X[va])
            report = classification_report(y[va], preds, output_dict=True, zero_division=0)
            p = report.get('1', {})
            logging.info(
                f"  Fold {fold+1}  "
                f"precision={p.get('precision', float('nan')):.3f}  "
                f"recall={p.get('recall', float('nan')):.3f}  "
                f"f1={p.get('f1-score', float('nan')):.3f}  "
                f"support={int(p.get('support', 0))}"
            )

        self.model.fit(X, y)
        self.is_fitted = True
        logging.info(f"\nRF trained on {len(y)} trades \n({y.sum()} profitable / {(1-y).sum()} not).")

        fi = pd.Series(
            self.model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)
        logging.info(f"\nFeature importances: {fi.round(4).to_string()}")

    def check_quality(self, feature_row: np.ndarray) -> float:
        if not self.is_fitted:
            return 1.0
        return float(self.model.predict_proba(feature_row.reshape(1, -1))[0, 1])


class XGBoostModel(QualityModelBase):
    """
    XGBoost classifier trained on the same feature set as RandomForestModel.
    Uses isotonic-regression calibration so gate_signal() can threshold on
    true probability rather than a raw score.
    """

    threshold_attr = "xgb_threshold"
    file_attr = "xgb_file"

    def __init__(self, cfg=None, n_splits: int = None):
        super().__init__(cfg=cfg, n_splits=n_splits)
        self._base = XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    def train(self, features: pd.DataFrame, labels: pd.Series):
        self.feature_cols = list(features.columns)
        self.feat_template = pd.Series(np.nan, index=self.feature_cols)

        mask = labels.notna()
        X = features.loc[mask].fillna(0).values
        y = labels.loc[mask].values.astype(int)

        if len(np.unique(y)) < 2:
            logging.warning("XGB: only one class in labels. Not trained.")
            return

        logging.info(f"\nXGB class distribution: {np.bincount(y)}")
        logging.info(f"XGB positive rate: {y.mean():.3f}")

        n_splits = max(2, min(self.n_splits, int(np.bincount(y).min())))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        logging.info("XGB Walk-forward CV:")
        for fold, (tr, va) in enumerate(tscv.split(X)):
            fold_counts = np.bincount(y[tr])
            if len(fold_counts) < 2 or fold_counts.min() < 2:
                logging.warning(f"XGB Fold {fold+1}: insufficient class support in training split, skipping.")
                continue
            calib_cv = int(min(3, fold_counts.min()))
            self.model = CalibratedClassifierCV(self._base, cv=calib_cv, method="isotonic")
            self.model.fit(X[tr], y[tr])
            preds = self.model.predict(X[va])
            report = classification_report(y[va], preds, output_dict=True, zero_division=0)
            p = report.get('1', {})
            logging.info(
                f"  Fold {fold+1}  "
                f"precision={p.get('precision', float('nan')):.3f}  "
                f"recall={p.get('recall', float('nan')):.3f}  "
                f"f1={p.get('f1-score', float('nan')):.3f}  "
                f"support={int(p.get('support', 0))}"
            )

        final_counts = np.bincount(y)
        if final_counts.min() < 2:
            logging.warning(
                f"XGB: minority class has only {final_counts.min()} sample(s). "
                f"Fitting without calibration."
            )
            self.model = self._base
        else:
            calib_cv = int(min(3, final_counts.min()))
            self.model = CalibratedClassifierCV(self._base, cv=calib_cv, method="isotonic")

        self.model.fit(X, y)
        self.is_fitted = True
        logging.info(f"XGB trained on {len(y)} trades ({y.sum()} profitable).")

        try:
            base_xgb = self.model.calibrated_classifiers_[0].estimator
        except AttributeError:
            base_xgb = self.model
        fi = pd.Series(
            base_xgb.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)
        logging.info(f"XGB Feature importances:\n{fi.round(4).to_string()}")
