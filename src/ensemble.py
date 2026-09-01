"""Combine LSTM and XGBoost predictions: naive average, a weight grid-search,
and a linear stacker -- each fit on one half of the test period and scored
on the other half, so the reported number isn't fit on the same data it's
evaluated on.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

from .metrics import wape


def naive_average(preds_a: np.ndarray, preds_b: np.ndarray) -> np.ndarray:
    return 0.5 * preds_a + 0.5 * preds_b


def grid_search_weight(preds_a: np.ndarray, preds_b: np.ndarray, actual: np.ndarray,
                        weights=None) -> tuple[float, float]:
    """Search blend weights `w * preds_a + (1 - w) * preds_b` and return the
    weight and WAPE that minimize WAPE against `actual` (all three arrays
    should be the *fit* slice only -- score the returned weight separately
    on a held-out slice).
    """
    if weights is None:
        weights = np.arange(0.0, 1.01, 0.05)
    best_w, best_wape = None, float("inf")
    for w in weights:
        blend = w * preds_a + (1 - w) * preds_b
        w_score = wape(actual, blend)
        if w_score < best_wape:
            best_wape, best_w = w_score, w
    return best_w, best_wape


def fit_linear_stacker(preds_a: np.ndarray, preds_b: np.ndarray, actual: np.ndarray) -> LinearRegression:
    """Fit a linear regression on [preds_a, preds_b] -> actual (the fit
    slice). Unlike the grid search, this also learns an intercept and
    isn't constrained to weights that sum to 1.
    """
    X = np.column_stack([preds_a, preds_b])
    return LinearRegression().fit(X, actual)


def evaluate_ensembles(lstm_preds: np.ndarray, xgb_preds: np.ndarray, actual: np.ndarray,
                        fit_frac: float = 0.5) -> dict:
    """Compare naive averaging, grid-searched weighting, and a linear
    stacker, all fit on the first `fit_frac` of the (already time-ordered)
    predictions and scored on the remainder.
    """
    n = len(actual)
    half = int(n * fit_frac)
    fit, score = slice(0, half), slice(half, None)

    best_w, fit_wape = grid_search_weight(xgb_preds[fit], lstm_preds[fit], actual[fit])
    grid_blend_score = best_w * xgb_preds[score] + (1 - best_w) * lstm_preds[score]

    stacker = fit_linear_stacker(lstm_preds[fit], xgb_preds[fit], actual[fit])
    stacker_preds_score = stacker.predict(np.column_stack([lstm_preds[score], xgb_preds[score]]))

    naive_score = naive_average(xgb_preds[score], lstm_preds[score])

    scores = {
        "naive_50_50": wape(actual[score], naive_score),
        "grid_weight": wape(actual[score], grid_blend_score),
        "linear_stacker": wape(actual[score], stacker_preds_score),
    }
    best_name = min(scores, key=scores.get)

    return {
        "scores": scores,
        "best": best_name,
        "grid_weight_value": best_w,
        "grid_weight_fit_wape": fit_wape,
        "stacker_coefs": {"lstm": float(stacker.coef_[0]), "xgb": float(stacker.coef_[1]),
                           "intercept": float(stacker.intercept_)},
    }
