"""Feature engineering: lag/rolling/Fourier features and LSTM sequence windows."""
import numpy as np
import pandas as pd

# The 13-feature baseline used throughout the single-series sections. Any
# variant that swaps in the corrected holiday flag replaces 'is_holiday'
# with 'is_holiday_v2' (see data.add_holiday_v2_and_shock) rather than
# redefining this list.
BASE_FEATURE_COLS = [
    "sales_lag1", "sales_lag7", "sales_lag14", "sales_lag28",
    "sales_roll7_mean", "sales_roll7_std",
    "onpromotion", "is_holiday", "oil_lag1",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
]


def engineer_features(df: pd.DataFrame, feat_cols_base: list[str] = BASE_FEATURE_COLS) -> pd.DataFrame:
    """Add lag/rolling-window sales features, day-of-week + month Fourier
    encodings, and a 1-day-lagged oil price, then drop rows that don't have
    a full feature window yet (early rows with no sales_lag28, etc.).
    """
    sx = df.copy().sort_values("date").reset_index(drop=True)
    sx["day_of_week"] = sx["date"].dt.dayofweek
    sx["month"] = sx["date"].dt.month
    sx["oil_lag1"] = sx["dcoilwtico"].shift(1)
    sx["sales_lag1"] = sx["sales"].shift(1)
    sx["sales_lag7"] = sx["sales"].shift(7)
    sx["sales_lag14"] = sx["sales"].shift(14)
    sx["sales_lag28"] = sx["sales"].shift(28)
    sx["sales_roll7_mean"] = sx["sales"].shift(1).rolling(7).mean()
    sx["sales_roll7_std"] = sx["sales"].shift(1).rolling(7).std()
    sx["dow_sin"] = np.sin(2 * np.pi * sx["day_of_week"] / 7)
    sx["dow_cos"] = np.cos(2 * np.pi * sx["day_of_week"] / 7)
    sx["month_sin"] = np.sin(2 * np.pi * sx["month"] / 12)
    sx["month_cos"] = np.cos(2 * np.pi * sx["month"] / 12)
    sx = sx.dropna(subset=feat_cols_base).reset_index(drop=True)
    return sx


def add_second_harmonic(df: pd.DataFrame) -> pd.DataFrame:
    """Add a second Fourier harmonic for day-of-week and month seasonality.

    Worth trying because a single sin/cos pair can only represent one
    harmonic -- XGBoost feature importances showed dow_sin carrying a
    disproportionate share of total importance, suggesting the weekly
    pattern isn't purely sinusoidal.
    """
    df = df.copy()
    df["dow_sin2"] = np.sin(4 * np.pi * df["day_of_week"] / 7)
    df["dow_cos2"] = np.cos(4 * np.pi * df["day_of_week"] / 7)
    df["month_sin2"] = np.sin(4 * np.pi * df["month"] / 12)
    df["month_cos2"] = np.cos(4 * np.pi * df["month"] / 12)
    return df


def add_promo_roll7(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 7-day rolling mean of on-promotion status, lagged by one day --
    a smoother signal than the same-day raw promo count.
    """
    df = df.copy()
    df["promo_roll7"] = df["onpromotion"].shift(1).rolling(7).mean()
    return df


def create_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int):
    """Slide a fixed-length window over (X, y) to build LSTM training
    sequences: each window of `sequence_length` rows predicts the target
    at the row immediately after it.
    """
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length])
    return np.array(X_sequences), np.array(y_sequences)
