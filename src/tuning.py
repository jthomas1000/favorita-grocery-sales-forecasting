"""Optuna hyperparameter search for the LSTM and both XGBoost models.

Each `make_*_objective` function builds an Optuna objective closed over the
data it needs, so the search space and the training loop it drives stay in
one place. Call `run_study` to actually execute it.
"""
import numpy as np
import optuna
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from .features import create_sequences
from .models import TunableLSTM, DEVICE
from .metrics import wape


def run_study(objective, n_trials: int, seed: int = 42, direction: str = "minimize") -> optuna.Study:
    """Run a seeded TPE study. Seeding the sampler (not just torch/numpy)
    is what makes the *search path* reproducible -- an unseeded sampler can
    land on different "best" hyperparameters across runs even with the same
    trial budget.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def make_lstm_objective(X_scaled: np.ndarray, y_scaled: np.ndarray, split_idx: int,
                         tune_epochs: int = 30, device: str = DEVICE):
    """LSTM search space: sequence length, hidden size, depth, dropout, LR.
    Each trial trains a fresh model for `tune_epochs` (short, for search
    speed) and scores it on the held-out split.
    """

    def objective(trial: optuna.Trial) -> float:
        seq_len = trial.suggest_categorical("sequence_length", [7, 14, 21, 28])
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
        seq_split = split_idx - seq_len
        if seq_split < 50:
            raise optuna.TrialPruned()
        X_train, X_val = X_seq[:seq_split], X_seq[seq_split:]
        y_train, y_val = y_seq[:seq_split], y_seq[seq_split:]

        X_train_t = torch.tensor(X_train).to(device)
        y_train_t = torch.tensor(y_train).to(device)
        X_val_t = torch.tensor(X_val).to(device)
        y_val_t = torch.tensor(y_val).to(device)

        model = TunableLSTM(X_train.shape[2], hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        for _epoch in range(tune_epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t), y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        return val_loss

    return objective


def make_xgb_objective(df, feat_cols: list[str], train_eval_fn):
    """Single-series XGBoost search space, scored by WAPE via `train_eval_fn`
    (pass `models.train_eval_xgb`).
    """

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            early_stopping_rounds=30, eval_metric="rmse", random_state=42,
        )
        return train_eval_fn(df, feat_cols, params)["wape"]

    return objective


def make_pooled_xgb_objective(X_train, y_train, X_test, y_test):
    """Pooled XGBoost search space: same hyperparameter ranges as the
    single-series search, plus the categorical-feature settings the pooled
    model needs (`enable_categorical`, `tree_method='hist'`).
    """

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            early_stopping_rounds=30, eval_metric="rmse", random_state=42,
            enable_categorical=True, tree_method="hist",
        )
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return wape(y_test, model.predict(X_test))

    return objective
