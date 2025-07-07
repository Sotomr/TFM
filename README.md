# TFM Portfolio Experiments

This repository evaluates whether adding a machine learning model improves a portfolio compared with a classical Markowitz approach.

## Research question

*¿Añadir un modelo ML mejora la cartera respecto al Markowitz clásico?*

## Workflow

1. **Data collection** (`notebooks/01_get_data.ipynb`)
   - Downloads price data and the VIX index【F:notebooks/01_get_data.ipynb†L120-L137】.
2. **Preprocessing**
   - For classical methods (Markowitz/Ridge) run `notebooks/02_preprocess_data.ipynb` to compute log returns and save scalers.
   - For the LSTM‑5d(+VIX) model use `notebooks/lstm5d/02_preprocess_lstm5d.ipynb` which creates 60‑day windows of returns and momentum shifted by one day to avoid look‑ahead bias【F:notebooks/lstm5d/02_preprocess_lstm5d.ipynb†L370-L382】.
   - Covariance matrices for Markowitz are generated with `notebooks/04_calc_covariance.ipynb` using Ledoit‑Wolf shrinkage【F:notebooks/04_calc_covariance.ipynb†L20-L74】 and saved to `data/processed/covariance_last60d.pkl`【F:notebooks/04_calc_covariance.ipynb†L78-L98】.
3. **Training**
   - The LSTM‑5d model is trained in `notebooks/lstm5d/03_train_lstm5d.ipynb`. Features and targets are scaled using `StandardScaler` fitted only on the training split and the scalers are stored for later use【F:notebooks/lstm5d/03_train_lstm5d.ipynb†L120-L143】.
   - Ridge regression models (one per asset) can be fitted with a similar train/validation split (see `notebooks/xgb/03_train_xgb.ipynb` for a template).
4. **Backtesting**
   - Execute `notebooks/06_backtest.ipynb` specifying `cfg.MODEL_TYPE` (`markowitz`, `ridge` or `lstm5d`) to generate portfolio weights. Input features are again shifted by one day to prevent look‑ahead bias【F:notebooks/06_backtest.ipynb†L78-L85】.
   - Results are saved under `results/` and compared in `notebooks/08_comparacion_modelos.ipynb`, which shows the performance metrics for each strategy【F:notebooks/08_comparacion_modelos.ipynb†L38-L59】.

## Avoiding look‑ahead bias

- Returns used as features are shifted by one day before forming the learning dataset【F:notebooks/lstm5d/02_preprocess_lstm5d.ipynb†L370-L381】【F:notebooks/06_backtest.ipynb†L78-L85】.
- When scaling, the `StandardScaler` is fitted only on the training data and later applied to validation and test sets【F:notebooks/lstm5d/03_train_lstm5d.ipynb†L120-L137】.

## Requirements and hardware

The notebooks rely on Python packages such as `pandas`, `numpy`, `scikit-learn`, `joblib`, `tensorflow` (for the LSTM), `lightgbm`, and `pymoo`. Training the LSTM‑5d network can be time‑consuming; a modern GPU is recommended but CPU execution is also possible (see GPU detection output in the training logs【F:notebooks/tft/03_train_tft.ipynb†L112-L112】).

## Running the Ridge backtest

1. Execute `notebooks/02_preprocess_data.ipynb` to compute daily returns and save the scalers.
2. Train a Ridge regression model for each asset using `notebooks/ridge/02_train_ridge.ipynb`.
3. Set `cfg.MODEL_TYPE = "ridge"` in `src/config.py` and run `notebooks/06_backtest.ipynb`.
4. The resulting returns and equity curve are stored in `results/backtest_ridge.pkl`【F:notebooks/06_backtest.ipynb†L318-L327】.

## LSTM‑5d+VIX workflow

1. Preprocess the data with `notebooks/lstm5d/02_preprocess_lstm5d.ipynb`. This notebook produces `data/processed/lstm5d_vix.pkl` containing the 60‑day sequences with VIX and momentum features【F:notebooks/lstm5d/02_preprocess_lstm5d.ipynb†L520-L534】.
2. Train the model in `notebooks/lstm5d/03_train_lstm5d.ipynb`; it loads `lstm5d_vix.pkl` and saves the network as `models/lstm5d_vix.keras` along with the training history【F:notebooks/lstm5d/03_train_lstm5d.ipynb†L32-L39】【F:notebooks/lstm5d/03_train_lstm5d.ipynb†L333-L343】.
3. Edit `src/config.py` so that `MODEL_TYPE = "lstm5d"` and `LSTM5D_MODEL_NAME = "lstm5d_vix.keras"` then run `notebooks/06_backtest.ipynb` to generate the trading signals.
4. The equity curve of this backtest is saved to `results/backtest_lstm5d.pkl`.

## Walk‑forward evaluation

`notebooks/09_walk_forward.ipynb` applies a walk‑forward procedure over several sub‑periods. After running it, the cumulative equity series can be inspected (and optionally saved) from the variable `equity` computed at the end of the notebook【F:notebooks/09_walk_forward.ipynb†L108-L129】.

