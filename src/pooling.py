"""Pool multiple series into one dataframe and fit a single global XGBoost
model across all of them, with store/family metadata as categorical
features so the model can borrow strength across series.
"""
import pandas as pd
from xgboost import XGBRegressor

from .metrics import wape

CAT_COLS = ["family", "city", "state", "store_type", "cluster", "store_nbr"]

POOLED_XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, early_stopping_rounds=30, eval_metric="rmse", random_state=42,
    enable_categorical=True, tree_method="hist",
)


def build_pooled_frame(series_frames: dict, stores: pd.DataFrame, train_frac: float = 0.8) -> pd.DataFrame:
    """Stack per-series feature frames into one pooled dataframe, tagging
    each row with its series' store/family metadata and a chronological
    train/test split *within that series* (so pooling doesn't let one
    series' future leak into another's training rows).

    `series_frames` maps (store_nbr, family) -> feature-engineered dataframe.
    """
    frames = []
    for (store_nbr, family), sx in series_frames.items():
        row = stores[stores["store_nbr"] == store_nbr].iloc[0]
        sx = sx.copy()
        sx["store_nbr"], sx["family"] = store_nbr, family
        sx["city"], sx["state"] = row["city"], row["state"]
        sx["store_type"], sx["cluster"] = row["type"], row["cluster"]
        split_i = int(len(sx) * train_frac)
        sx["_split"] = ["train"] * split_i + ["test"] * (len(sx) - split_i)
        frames.append(sx)

    pooled = pd.concat(frames, ignore_index=True)
    for c in CAT_COLS:
        pooled[c] = pooled[c].astype("category")
    return pooled


def fit_pooled_xgb(pooled_df: pd.DataFrame, feat_cols: list[str], params: dict | None = None):
    """Fit one XGBoost model across every series in `pooled_df` (train rows
    only) and score it per-series and overall on the test rows.

    Returns (model, pooled_df_with_predictions, per_series_wape_df, overall_wape).
    """
    feat_cols_full = feat_cols + CAT_COLS
    train_mask = pooled_df["_split"] == "train"
    test_mask = pooled_df["_split"] == "test"

    X_train = pooled_df.loc[train_mask, feat_cols_full]
    y_train = pooled_df.loc[train_mask, "sales"].values.astype("float32")
    X_test = pooled_df.loc[test_mask, feat_cols_full]
    y_test = pooled_df.loc[test_mask, "sales"].values.astype("float32")

    model = XGBRegressor(**(params or POOLED_XGB_PARAMS))
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pooled_df = pooled_df.copy()
    pooled_df.loc[test_mask, "pred"] = model.predict(X_test)

    per_series = []
    for (store_nbr, family), _ in pooled_df.groupby(["store_nbr", "family"]):
        m = (pooled_df["store_nbr"] == store_nbr) & (pooled_df["family"] == family) & test_mask
        actual, pred = pooled_df.loc[m, "sales"].values, pooled_df.loc[m, "pred"].values
        per_series.append({"store_nbr": store_nbr, "family": family, "wape": wape(actual, pred)})
    per_series_df = pd.DataFrame(per_series)

    overall_wape = wape(y_test, pooled_df.loc[test_mask, "pred"].values)
    return model, pooled_df, per_series_df, overall_wape
