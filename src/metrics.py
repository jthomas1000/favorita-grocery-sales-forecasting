"""Evaluation metrics."""
import numpy as np


def wape(actual, pred) -> float:
    """Weighted Absolute Percentage Error, as a percentage.

    Sums absolute errors and absolute actuals separately before dividing,
    so it doesn't blow up on near-zero individual days the way plain MAPE
    does -- the standard choice for intermittent retail demand.
    """
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    return float((np.abs(actual - pred).sum() / np.abs(actual).sum()) * 100)
