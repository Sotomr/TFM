# TFM Portfolio Optimization with Machine Learning Models

This repository implements a comprehensive portfolio optimization system that combines multiple machine learning architectures with evolutionary algorithms (NSGA-II) to evaluate whether advanced ML models can outperform classical Markowitz optimization in quantitative asset management.

## Research question

*¿Los modelos de machine learning avanzados pueden superar consistentemente al enfoque clásico de Markowitz y al benchmark S&P 500 en la gestión cuantitativa de portafolios?*

## System Overview

**Portfolio Framework**: Multi-objective optimization using NSGA-II evolutionary algorithm with risk minimization and return maximization objectives. The system manages a universe of 40 assets (38 traditional stocks + BTC + ETH) with realistic constraints including position limits (≤20%), crypto allocation (≤10%), and transaction costs (0.2% per trade).

**ML Models Evaluated**: Six distinct architectures ranging from classical linear methods to advanced neural networks, each with specialized feature engineering and temporal modeling approaches.

## Workflow

### 1. **Data Collection** (`notebooks/01_get_data.ipynb`)
   - Downloads daily price data for 40 large-cap assets plus VIX index from 2019-2025
   - Includes traditional stocks (AAPL, MSFT, GOOGL, etc.) and cryptocurrencies (BTC-USD, ETH-USD)
   - Federal funds rate data for macroeconomic features

### 2. **Preprocessing Pipeline**
   
   **General preprocessing**:
   - `notebooks/02_preprocess_data.ipynb` - Computes log returns and saves scalers for classical methods
   - `notebooks/04_calc_covariance.ipynb` - Generates covariance matrices using Ledoit-Wolf shrinkage estimator
   
   **Model-specific preprocessing**:
   - `notebooks/lstm5d/02_preprocess_lstm5d.ipynb` - Creates 60-day sequences with returns, momentum, and VIX features (81 total features) with 1-day shift to prevent look-ahead bias
   - `notebooks/CNN5d/02_preprocess_cnn5d.ipynb` - Temporal windows for convolutional processing with 5-day prediction targets
   - `notebooks/gru5d/02_preprocess_gru5d.ipynb` - GRU-specific feature engineering with momentum indicators
   - `notebooks/xgb/02_preprocess_xgb_enhanced.ipynb` - Advanced technical features including RSI, Bollinger bands, cross-correlations, and macro indicators (24 features total)
   - `notebooks/ridge/02_preprocess_ridge.ipynb` - Linear model feature preparation

### 3. **Model Training**
   
   **Neural Networks**:
   - `notebooks/lstm/03_train_lstm.ipynb` - LSTM-1d with 40-day return sequences predicting 1-day returns
   - `notebooks/lstm5d/03_train_lstm5d.ipynb` - Advanced LSTM with VIX integration and momentum features
   - `notebooks/CNN5d/03_train_cnn5d.ipynb` - Convolutional neural network for temporal pattern recognition
   - `notebooks/gru5d/03_train_gru5d.ipynb` - Gated recurrent units with 5-day prediction horizon
   
   **Tree-based Models**:
   - `notebooks/xgb/03_train_xgb_enhanced.ipynb` - XGBoost with 24 engineered financial features
   
   **Linear Models**:
   - `notebooks/ridge/02_train_ridge.ipynb` - Ridge regression with L2 regularization (one model per asset)

### 4. **Walk-Forward Backtesting** (`notebooks/06_backtest.ipynb`)
   - Configure model type in `src/config.py` (options: `lstm_t1`, `lstm5d`, `cnn5d`, `gru5d`, `xgb_enhanced`, `ridge`)
   - Executes walk-forward evaluation with 10-day rebalancing frequency
   - Generates predictions with strict temporal ordering to prevent data leakage
   - Applies NSGA-II optimization with realistic constraints and transaction costs
   - Results saved to `results/backtest_{model_type}.pkl`

### 5. **Performance Analysis and Visualization**
   
   **Individual model analysis**:
   - `notebooks/{model}/07_visualizaciones_{model}.ipynb` - Model-specific performance dashboards with equity curves, drawdown analysis, and regime-based evaluation
   
   **Comparative analysis**:
   - `notebooks/08_comparacion_modelos.ipynb` - Cross-model performance comparison with correlation analysis and statistical significance testing
   - `notebooks/09_walk_forward.ipynb` - Temporal stability analysis across multiple sub-periods
   - `notebooks/10_compare_methods.ipynb` - Comprehensive statistical comparison framework

## Model Architectures

| Model | Type | Input Features | Target | Key Innovation |
|-------|------|----------------|--------|----------------|
| **LSTM-1d** | RNN | 40-day return sequences | 1-day returns | Temporal pattern recognition |
| **LSTM5D** | RNN | Returns + Momentum + VIX (81 features) | 1-day returns | Macro-integrated predictions |
| **CNN5D** | CNN | 80 temporal features | 5-day returns | Convolutional pattern detection |
| **GRU5D** | RNN | Returns + Momentum (80 features) | 5-day returns | Gated temporal modeling |
| **XGBoost Enhanced** | Gradient Boosting | 24 technical indicators | 5-day returns | Advanced feature engineering |
| **Ridge** | Linear | Statistical features | Variable horizon | Regularized linear baseline |

## Bias Prevention and Validation

**Look-ahead bias prevention**:
- All features shifted by minimum 1 day before dataset creation
- Strict chronological splits: training → validation → test
- No future information used in predictions or optimization

**Data leakage controls**:
- Scalers fitted exclusively on training data, applied to validation/test
- Model selection based on validation performance (test set never seen during development)
- Features computed using only historically available information at prediction time

**Overfitting mitigation**:
- Early stopping with validation loss monitoring for neural networks
- Walk-forward evaluation across multiple out-of-sample periods
- Statistical significance testing for performance differences

## Key Results Summary

**Performance Overview (2019-2025)**:
- **Best Performer**: LSTM-1d with 94.51% net return (14.86% annual) and 0.925 Sharpe ratio
- **Benchmark**: S&P 500 achieved 120.73% return (17.93% annual) with 0.858 Sharpe ratio
- **Underperformance**: All models underperform SPY primarily due to transaction costs and conservative optimization
- **Risk Management**: Superior volatility control (15-18% vs 21% SPY) and drawdown management

**Critical Findings**:
- Transaction costs (0.2% per trade) create 13-21% annual drag on performance
- Conservative optimization (NSGA-II) generates overly diversified portfolios
- ML models provide superior risk-adjusted returns but fail to overcome implementation costs
- Best strategy combines simplest architecture (LSTM-1d) with effective risk management

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
# or
python install_dependencies.py
```

### Complete Pipeline Execution

1. **Download data and setup**:
   ```bash
   jupyter notebook notebooks/01_get_data.ipynb
   ```

2. **Choose model and preprocess** (example with LSTM5D):
   ```bash
   jupyter notebook notebooks/lstm5d/02_preprocess_lstm5d.ipynb
   jupyter notebook notebooks/lstm5d/03_train_lstm5d.ipynb
   ```

3. **Configure and execute backtest**:
   ```python
   # Edit src/config.py
   MODEL_TYPE = "lstm5d"
   LSTM5D_MODEL_NAME = "lstm5d.keras"
   ```
   ```bash
   jupyter notebook notebooks/06_backtest.ipynb
   ```

4. **Analyze results**:
   ```bash
   jupyter notebook notebooks/lstm5d/07_visualizaciones_lstm5d.ipynb
   jupyter notebook notebooks/08_comparacion_modelos.ipynb
   ```

## Configuration

**Core parameters** (`src/config.py`):
```python
MODEL_TYPE = "lstm5d"     # Model selection
REBAL_FREQ = 10          # Rebalancing frequency (days)
W_MAX = 0.20             # Maximum position size
CRYPTO_MAX = 0.10        # Maximum crypto allocation  
COST_TRADE = 0.002       # Transaction cost per trade
WINDOW = 60              # Feature lookback window
```

## Requirements

**Software**: Python 3.8+, TensorFlow/Keras, XGBoost, PyMOO, yfinance, scikit-learn, pandas, numpy

**Hardware**: Multi-core CPU (optimization intensive), 8GB+ RAM, optional GPU for neural network training

**Execution time**: 5 minutes (data) + 15 minutes (preprocessing) + 30 minutes to 2 hours (training) + 45 minutes (backtesting)

## Repository Structure
