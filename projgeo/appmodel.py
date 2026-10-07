"""The demo app's "how likely is this AI-generated" score.

A random forest on the geometry-only residual vector of `explain.FEATURES`,
calibrated by cross-validated Platt scaling and then re-based to a 50/50
prior: the training set is mostly generated images, so its raw calibrated
probability would carry that imbalance into every answer.  The re-based
number answers "if this image were equally likely to be real or generated
before we looked, how likely is it generated given its geometry?"
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .explain import FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "app_model.joblib"
# The app measures each image at its own width, capped at 1024 px and never enlarged.  The report's
# group comparisons use 640 px to match York Urban; for scoring single images the extra line
# detail matters more: cross-validated AUC 0.77 at 640 px, 0.81 with this cap, and image width
# alone scores 0.52, so the gain is not a resolution shortcut.
APP_MAX_WIDTH = 1024


def analysis_width(img) -> int:
    return min(img.shape[1], APP_MAX_WIDTH)


def make_model(seed: int = 0):
    forest = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                       ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                     class_weight="balanced", random_state=seed,
                                                     n_jobs=-1))])
    # ensemble=False: Platt scaling fitted on 5-fold out-of-fold predictions, one
    # forest stored (keeps the model file small enough to commit)
    return CalibratedClassifierCV(forest, method="sigmoid", cv=5, ensemble=False)


def rebase(p, train_pos_share: float, target_pos_share: float = 0.5):
    """Move calibrated probabilities from the training prior to a target prior."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * (target_pos_share / (1 - target_pos_share)) * \
        ((1 - train_pos_share) / train_pos_share)
    return odds / (1 + odds)


def feature_matrix(rows) -> np.ndarray:
    """rows: iterable of dicts (or a DataFrame) with the FEATURES keys."""
    if hasattr(rows, "loc"):
        return rows[FEATURES].astype(float).values
    return np.array([[np.nan if r.get(k) is None else r.get(k) for k in FEATURES] for r in rows], float)


def load(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def ai_probability(bundle: dict, features: dict) -> float:
    raw = bundle["model"].predict_proba(feature_matrix([features]))[0, 1]
    return float(rebase(raw, bundle["train_ai_share"]))


def percentile_vs(bundle: dict, feature: str, value: float, group: str = "real") -> float:
    """Share of `group` training images with a value below `value` (0-100)."""
    ref = np.asarray(bundle["reference"][group][feature], float)
    ref = ref[np.isfinite(ref)]
    if not np.isfinite(value) or len(ref) == 0:
        return float("nan")
    return float(100 * np.mean(ref < value))
