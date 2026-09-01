"""Walk-forward (rolling-origin) backtesting.

A single chronological train/test split can land on an easy or hard slice
of time by luck. These helpers instead retrain across several expanding
windows and report the spread, not just the mean.
"""
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import create_sequences
from .models import TunableLSTM, LSTMConfig, DEVICE
from .metrics import wape
from .pooling import CAT_COLS, POOLED_XGB_PARAMS


def walk_forward_lstm(df, feat_cols: list[str], config: LSTMConfig, k_folds: int = 5,
                       epochs: int = 200, patience: int = 25, seed: int = 42,
                       device: str = DEVICE) -> pd.DataFrame:
    """Expanding-window backtest for a single series: fold k trains on all
    rows up to a cutoff and tests on the next block, with cutoffs marching
    forward through time. Returns one row per fold with train/test size and WAPE.
    """
    n_rows = len(df)
    usable = n_rows - config.sequence_length
    fold_size = usable // (k_folds + 1)

    Xr = df[feat_cols].values.astype("float32")
    yr = df["sales"].values.astype("float32").reshape(-1, 1)

    results = []
    for k in range(k_folds):
        train_end = config.sequence_length + fold_size * (k + 1)
        test_end = min(train_end + fold_size, n_rows)
        if test_end <= train_end:
            continue

        x_scaler = StandardScaler().fit(Xr[:train_end])
        y_scaler = StandardScaler().fit(yr[:train_end])
        X_scaled, y_scaled = x_scaler.transform(Xr), y_scaler.transform(yr)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, config.sequence_length)
        train_seq_end = train_end - config.sequence_length
        test_seq_end = test_end - config.sequence_length
        X_train, X_test = X_seq[:train_seq_end], X_seq[train_seq_end:test_seq_end]
        y_train, y_test = y_seq[:train_seq_end], y_seq[train_seq_end:test_seq_end]
        if len(X_test) == 0 or len(X_train) < 50:
            continue

        torch.manual_seed(seed)
        model = TunableLSTM(X_train.shape[2], config.hidden_size, config.num_layers, config.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        X_train_t, y_train_t = torch.tensor(X_train).to(device), torch.tensor(y_train).to(device)
        X_test_t, y_test_t = torch.tensor(X_test).to(device), torch.tensor(y_test).to(device)

        best_val, bad, best_state = float("inf"), 0, None
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
                best_val, bad = val_loss, 0
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test_t).cpu().numpy()
        preds = y_scaler.inverse_transform(preds_scaled).flatten()
        actual = y_scaler.inverse_transform(y_test).flatten()
        results.append({
            "fold": k + 1, "train_rows": train_end, "test_rows": test_end - train_end,
            "wape": wape(actual, preds),
        })

    return pd.DataFrame(results)


def walk_forward_pooled_xgb(series_frames: dict, feat_cols: list[str], n_folds: int = 5,
                             params: dict | None = None) -> pd.DataFrame:
    """Expanding-window backtest for the pooled model. `TimeSeriesSplit` is
    applied independently to each series (on its own row count, so no
    series' fold boundary leaks into another's calendar), then folds are
    pooled fold-by-fold: fold k trains one pooled model across all series'
    fold-k training rows and scores it on all series' fold-k test rows at once.

    `series_frames` maps (store_nbr, family) -> feature-engineered dataframe
    (already carrying store_nbr/family/city/state/store_type/cluster columns,
    as produced by `pooling.build_pooled_frame` before the '_split' column
    is added -- pass each series' frame with '_split' dropped).
    """
    feat_cols_full = feat_cols + CAT_COLS
    tscv = TimeSeriesSplit(n_splits=n_folds)
    fold_splits = {key: list(tscv.split(frame)) for key, frame in series_frames.items()}

    results = []
    for fold_idx in range(n_folds):
        train_parts, test_parts = [], []
        for key, frame in series_frames.items():
            tr_idx, te_idx = fold_splits[key][fold_idx]
            train_parts.append(frame.iloc[tr_idx])
            test_parts.append(frame.iloc[te_idx])
        fold_train = pd.concat(train_parts, ignore_index=True)
        fold_test = pd.concat(test_parts, ignore_index=True)

        for c in CAT_COLS:
            fold_train[c] = fold_train[c].astype("category")
            fold_test[c] = pd.Categorical(fold_test[c], categories=fold_train[c].cat.categories)

        X_train = fold_train[feat_cols_full]
        y_train = fold_train["sales"].values.astype("float32")
        X_test = fold_test[feat_cols_full]
        y_test = fold_test["sales"].values.astype("float32")

        model = XGBRegressor(**(params or POOLED_XGB_PARAMS))
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)

        results.append({
            "fold": fold_idx + 1, "n_train": len(fold_train), "n_test": len(fold_test),
            "wape": wape(y_test, preds),
        })

    return pd.DataFrame(results)
