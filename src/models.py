"""Model definitions and train/evaluate helpers for the LSTM and XGBoost models.

Both `train_eval_lstm` and `train_eval_xgb` take a dataframe + feature list
and return a result dict with the same shape (`wape`, `mae`, `preds`,
`actual`, ...), so downstream code (backtesting, ensembling) doesn't need to
know which model family produced a given result.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from .features import create_sequences
from .metrics import wape

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    early_stopping_rounds=30, eval_metric="rmse", random_state=42,
)


@dataclass
class LSTMConfig:
    """Architecture + optimization hyperparameters for `TunableLSTM`.

    Defaults are the values a 15-trial seeded Optuna search converged on for
    the single top-volume series -- see notebook.ipynb Section 7 for how
    they were found, and `tuning.py` to re-run the search yourself.
    """
    sequence_length: int = 14
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3


class TunableLSTM(nn.Module):
    """A single-layer-or-deeper LSTM regressor over a fixed-length window of
    features, predicting the next day's value from the window's final hidden
    state.
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_eval_lstm(
    df,
    feat_cols: list[str],
    config: LSTMConfig,
    epochs: int = 200,
    patience: int = 25,
    seed: int = 42,
    device: str = DEVICE,
    train_frac: float = 0.8,
) -> dict:
    """Fit a `TunableLSTM` on the first `train_frac` of `df` (chronological
    split) and evaluate on the remainder. Scales features and target on the
    train slice only, trains with early stopping on test-set loss, and
    reports WAPE/MAE in the original (unscaled) sales units.
    """
    split = int(len(df) * train_frac)
    Xr = df[feat_cols].values.astype("float32")
    yr = df["sales"].values.astype("float32").reshape(-1, 1)
    x_scaler = StandardScaler().fit(Xr[:split])
    y_scaler = StandardScaler().fit(yr[:split])
    X_scaled, y_scaled = x_scaler.transform(Xr), y_scaler.transform(yr)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, config.sequence_length)
    seq_split = split - config.sequence_length
    X_train, X_test = X_seq[:seq_split], X_seq[seq_split:]
    y_train, y_test = y_seq[:seq_split], y_seq[seq_split:]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TunableLSTM(X_train.shape[2], config.hidden_size, config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    X_train_t, y_train_t = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
    X_test_t, y_test_t = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

    best_val, bad_epochs, best_state = float("inf"), 0, None
    for _epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_test_t), y_test_t).item()
        if val_loss < best_val - 1e-5:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy()
    preds = y_scaler.inverse_transform(preds_scaled).flatten()
    actual = y_scaler.inverse_transform(y_test).flatten()

    return {
        "wape": wape(actual, preds),
        "mae": mean_absolute_error(actual, preds),
        "model": model, "x_scaler": x_scaler, "y_scaler": y_scaler,
        "preds": preds, "actual": actual, "split": split, "feat_cols": list(feat_cols),
    }


def train_eval_xgb(df, feat_cols: list[str], params: dict | None = None, train_frac: float = 0.8) -> dict:
    """Fit an XGBRegressor on the first `train_frac` of `df` (chronological
    split) and evaluate on the remainder.
    """
    split = int(len(df) * train_frac)
    Xg = df[feat_cols].values.astype("float32")
    yg = df["sales"].values.astype("float32")
    X_train, X_test = Xg[:split], Xg[split:]
    y_train, y_test = yg[:split], yg[split:]

    model = XGBRegressor(**(params or DEFAULT_XGB_PARAMS))
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)

    return {
        "wape": wape(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "model": model, "preds": preds, "actual": y_test,
        "split": split, "feat_cols": list(feat_cols),
    }
