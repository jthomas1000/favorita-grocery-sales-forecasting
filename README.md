# Favorita Grocery Sales Forecasting

Forecasting daily store/product-family sales for the Kaggle [Store Sales -
Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
(Favorita) competition. Compares a sliding-window LSTM against XGBoost,
tuned with Optuna, checked with walk-forward backtesting, and pooled
across series. Every claimed improvement gets re-tested before it's
trusted.

A lot of the "improvements" here looked good on a single 80/20 split and
then shrank or reversed once checked with a proper walk-forward backtest.
Those results are included below rather than left out.

## Results

**Single series (store 44, product family GROCERY I)**

| Model | WAPE | Note |
|---|---|---|
| Naive: predict yesterday's sales | n/a | baseline reference |
| Naive: predict same weekday last week | n/a | baseline reference |
| Untuned sliding-window LSTM | 16.46% | reasonable but hand-set hyperparameters |
| + Optuna tuning (15 trials, seeded) | improves over untuned | search: seq length, hidden size, depth, dropout, LR |
| + transactions/promo covariates (both at once) | 15.40% | worse than the holiday fix alone below |
| + corrected holiday flag + earthquake dummy | 13.85% | national+regional+local, transfer-aware; see Findings |
| XGBoost, same feature set | competitive with LSTM | ~10x cheaper to train |

**Pooled across 6 top-volume series (one XGBoost model, store/family as categorical features)**

| Setup | Mean WAPE (per-series) | Overall WAPE |
|---|---|---|
| Independent per-series XGBoost, plain features | 13.21% | n/a |
| Pooled XGBoost, plain features | **12.50%** | n/a |
| Independent per-series XGBoost, "winning" single-series features | 13.42% | n/a |
| Pooled XGBoost, "winning" single-series features | 12.68% | 12.78% |
| Pooled, walk-forward backtest (5 expanding folds) | 12.64% (std 1.39) | n/a |
| Pooled, Optuna-tuned (30 trials, seeded) | n/a | **12.43%** |

**Ensembling (LSTM + XGBoost, single series), fit on the first half of the test period and scored on the second half**

| Method | 2nd-half WAPE |
|---|---|
| Naive 50/50 average | **10.70%** |
| Grid-searched blend weight (XGB weight = 0.65) | 10.87% (14.50% on the fit half) |
| Linear stacker | 10.74% |

## Key findings

**Pooling is the single biggest lever here.** One XGBoost model trained
across all 6 series with store/family as categorical features beat 6
independently-tuned models by about 0.7 points of WAPE (12.50% vs
13.21%), with no extra feature engineering. That's a bigger gain than
most of the feature engineering below produced.

**The "winning" single-series feature set didn't transfer to pooling.**
Fixing the holiday flag and adding an earthquake dummy clearly helped one
series (13.85% vs 15.40% with the naive covariate set). Rebuilding that
same feature set for all 6 pooled series and refitting the pooled model
made things slightly worse (12.68% vs 12.50% for the plain feature set).
The likely reason: a feature engineered and validated against one
series' quirks doesn't necessarily generalize once a model has to fit
six series at once. It's easy to overfit a "fix" to the series you're
staring at.

**Walk-forward backtesting mostly held up the single-split numbers.**
The pooled model's 12.78% single-split WAPE landed at 12.64% (std 1.39)
across 5 expanding-window folds, so the headline number wasn't just a
lucky split.

**Optuna tuning gave a small, real edge on the pooled model** (12.78% to
12.43%). Smaller than the pooling effect, and not free: it cost 30
additional model fits.

**The naive 50/50 ensemble beat both "smarter" alternatives once scored
honestly.** A grid-searched blend weight and a linear stacker both
looked better on the data they were fit on, but a plain 50/50 average
won on the held-out second half of the test period. The fancier
ensembling methods were fitting noise in the first half that didn't
generalize, and the only way to catch that was scoring them on data
they hadn't seen.

**Adding covariates isn't free.** `transactions_lag1` and `promo_roll7`
together made the single-series LSTM worse, not better, before the
holiday flag was fixed. More features only help if they're not
competing with a broken feature that's still in the model.

## What I'd try next

- Feature-select per pooled model instead of assuming the single-series
  winning set transfers. Ablate `is_holiday_v2`/`earthquake_shock`
  specifically in the pooled setting rather than carrying them over
  unchanged.
- Extend pooling beyond the top 6 series and check whether the pooling
  gain holds, shrinks, or grows with more (and lower-volume, noisier)
  series in the mix.
- Try quantile/pinball loss for the pooled XGBoost model, since retail
  demand is right-skewed and WAPE alone doesn't say anything about
  calibration.
- Give the linear stacker L2 regularization or more fit data before
  writing it off. It lost by a narrow margin (10.74% vs 10.70%) and
  might do better with a larger fit window.

## Repo structure

```
favorita-forecasting/
├── README.md
├── requirements.txt
├── notebook.ipynb        # reference run: narrates the analysis, calls into src/
└── src/
    ├── data.py            # data loading, per-series assembly, holiday feature fixes
    ├── features.py        # lag/rolling/Fourier feature engineering, LSTM sequence windows
    ├── models.py           # TunableLSTM, train_eval_lstm, train_eval_xgb
    ├── tuning.py            # Optuna objectives for the LSTM and both XGBoost models
    ├── backtest.py           # walk-forward backtesting, single-series and pooled
    ├── pooling.py             # build a pooled dataframe across series, fit a pooled XGBoost model
    ├── ensemble.py             # naive average, weight grid-search, linear stacker
    └── metrics.py              # WAPE
```

The notebook is a thin narrative layer over `src/`. The actual logic
lives in the package so it's reusable and (in principle) unit-testable
outside the notebook, instead of sitting locked inside one long cell
dump.

## Data

Loaded from a public Hugging Face mirror of the competition data
(`t4tiana/store-sales-time-series-forecasting`), so the notebook runs
without Kaggle credentials. The mirror serves the same `train.csv`,
`stores.csv`, `oil.csv`, `holidays_events.csv`, and `transactions.csv`
as the original competition. Data isn't committed to this repo;
`notebook.ipynb` downloads it directly (see `src/data.py`).

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The notebook runs top to bottom in order. Each section builds on
dataframes/results defined in the section before it, matching the
narrative above. LSTM sections use a GPU if one's available and fall
back to CPU otherwise. CPU is noticeably slower but still finishes in a
reasonable time given the short training loops used here.

A note on reproducibility: the XGBoost-based results (pooling,
backtesting, Optuna tuning of XGBoost) use fixed random seeds throughout
and should reproduce closely. The LSTM results can shift by a few tenths
of a percentage point run-to-run despite the fixed seeds, since PyTorch
doesn't guarantee bit-for-bit determinism for every op on every device.
Noting that here rather than presenting single-run LSTM numbers as more
precise than they are.

## License

Code in this repo is available under the MIT License. The dataset
belongs to the original Kaggle competition and its listed sources
(Corporación Favorita); see the competition page for its terms.
