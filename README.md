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

