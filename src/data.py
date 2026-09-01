"""Data loading and per-series assembly for the Favorita dataset.

Source: Kaggle "Store Sales - Time Series Forecasting" competition,
mirrored (no credentials required) on Hugging Face at
https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting
"""
import pandas as pd

BASE_URL = "https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting/resolve/main/"

# The ~3-week demand shock that followed the April 16, 2016 Ecuador earthquake --
# a well-documented feature of this specific competition/dataset.
EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-07")


def load_raw(base_url: str = BASE_URL) -> dict:
    """Load train/stores/oil/holidays (required) and transactions (optional,
    not every mirror serves it -- callers should check for None).
    """
    train = pd.read_csv(base_url + "train.csv", parse_dates=["date"])
    stores = pd.read_csv(base_url + "stores.csv")
    oil = pd.read_csv(base_url + "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(base_url + "holidays_events.csv", parse_dates=["date"])
    try:
        transactions = pd.read_csv(base_url + "transactions.csv", parse_dates=["date"])
    except Exception:
        transactions = None
    return {
        "train": train,
        "stores": stores,
        "oil": oil,
        "holidays": holidays,
        "transactions": transactions,
    }


def top_volume_series(train: pd.DataFrame, n: int = 6):
    """Rank (store_nbr, family) pairs by total sales, descending.

    Returns the ranked Series (indexed by (store_nbr, family)) and the top
    `n` (store_nbr, family) tuples, so callers can grab the single busiest
    series or a pooled list of several.
    """
    vol = train.groupby(["store_nbr", "family"])["sales"].sum().sort_values(ascending=False)
    return vol, list(vol.index[:n])


def national_holiday_dates(holidays: pd.DataFrame) -> pd.Series:
    """The naive baseline holiday flag: any non-transferred National event.

    This is what the original baseline (single national flag) used. It
    misses regional/local holidays and mishandles transferred ones -- see
    `corrected_holiday_dates` for the per-store fix.
    """
    nat = holidays[(holidays["locale"] == "National") & (holidays["transferred"] == False)]
    return nat[["date"]].drop_duplicates()["date"]


def corrected_holiday_dates(holidays: pd.DataFrame, city: str, state: str) -> pd.Series:
    """Per-store holiday dates: National, plus Regional holidays matching the
    store's state, plus Local holidays matching the store's city -- and
    excluding transferred holidays and 'Work Day' rows (a holiday moved to
    another date is not a day off on its original date).
    """
    non_working = holidays[(holidays["transferred"] == False) & (holidays["type"] != "Work Day")]
    relevant = non_working[
        (non_working["locale"] == "National")
        | ((non_working["locale"] == "Regional") & (non_working["locale_name"] == state))
        | ((non_working["locale"] == "Local") & (non_working["locale_name"] == city))
    ]
    return relevant["date"].drop_duplicates()


def build_series(train: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame,
                  store_nbr: int, family: str) -> pd.DataFrame:
    """Assemble one store/family daily series: raw sales rows, densified to
    every calendar day in range, with oil price forward/back-filled in
    (oil.csv only has trading days) and the naive national-only holiday flag
    attached. Per-store holiday correction and lag/rolling/Fourier features
    are added separately -- see `add_holiday_v2_and_shock` and
    `features.engineer_features`.
    """
    ser = train[(train["store_nbr"] == store_nbr) & (train["family"] == family)].copy()
    ser = ser.sort_values("date").reset_index(drop=True)

    full_dates = pd.DataFrame({"date": pd.date_range(ser["date"].min(), ser["date"].max(), freq="D")})
    oil_full = full_dates.merge(oil, on="date", how="left")
    oil_full["dcoilwtico"] = oil_full["dcoilwtico"].ffill().bfill()
    ser = ser.merge(oil_full, on="date", how="left")

    nat_hol = national_holiday_dates(holidays)
    ser["is_holiday"] = ser["date"].isin(nat_hol).astype("float32")
    return ser


def add_holiday_v2_and_shock(df: pd.DataFrame, holidays: pd.DataFrame, city: str, state: str) -> pd.DataFrame:
    """Add the corrected `is_holiday_v2` flag and the `earthquake_shock` dummy
    for one store's city/state. Applied per-series (using that store's own
    city/state) so pooling doesn't leak one series' holiday calendar into
    another's.
    """
    df = df.copy()
    dates_v2 = corrected_holiday_dates(holidays, city, state)
    df["is_holiday_v2"] = df["date"].isin(dates_v2).astype("float32")
    df["earthquake_shock"] = ((df["date"] >= EARTHQUAKE_START) & (df["date"] <= EARTHQUAKE_END)).astype("float32")
    return df


def add_transactions_lag1(df: pd.DataFrame, transactions: pd.DataFrame, store_nbr: int) -> pd.DataFrame:
    """Merge in that store's daily transaction count, lagged by one day (the
    same-day count is only observed after the fact and would leak).
    """
    df = df.copy()
    txn_store = transactions[transactions["store_nbr"] == store_nbr][["date", "transactions"]]
    df = df.merge(txn_store, on="date", how="left")
    df["transactions"] = df["transactions"].ffill()
    df["transactions_lag1"] = df["transactions"].shift(1)
    return df
