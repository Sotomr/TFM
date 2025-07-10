import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import timedelta
from src import config as cfg

def calculate_rsi(prices, window=14):
    """Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_position(prices, window=20, num_std=2):
    """Posición dentro de las Bollinger Bands"""
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    position = (prices - lower_band) / (upper_band - lower_band)
    return position.clip(0, 1)

def predict_xgb_enhanced(df_prices: pd.DataFrame, fecha: pd.Timestamp, tickers: list[str]) -> pd.Series:
    """
    Predice retornos esperados a 5 días usando XGBoost Enhanced con 23+ features técnicas.
    
    Parámetros:
    - df_prices: DataFrame con precios (índice: fechas, columnas: tickers)
    - fecha: fecha en la que se quiere predecir (usa t-1 para features)
    - tickers: lista de tickers a procesar

    Devuelve:
    - Serie con retornos esperados a 5 días, indexada por ticker
    """
    # Cargar modelos enhanced
    modelos = joblib.load(cfg.DATA / "processed" / "xgb_enhanced_model.pkl")
    
    # Validaciones
    if fecha not in df_prices.index:
        raise ValueError(f"La fecha {fecha} no está en el índice del dataframe")
    
    idx = df_prices.index.get_loc(fecha)
    if idx < 200:  # Necesitamos más historia para features técnicas
        raise ValueError("No hay suficientes días anteriores para calcular features técnicas")

    # Descargar datos macro si no están en cache
    try:
        # ✅ VIX - FIX YFINANCE COLUMNS
        vix_raw = yf.download("^VIX", start=df_prices.index.min(), end=fecha + timedelta(days=1), progress=False, auto_adjust=False)
        if 'Adj Close' in vix_raw.columns:
            vix = vix_raw["Adj Close"].reindex(df_prices.index, method="ffill")
        elif 'Close' in vix_raw.columns:
            vix = vix_raw["Close"].reindex(df_prices.index, method="ffill")
        else:
            vix = vix_raw.iloc[:, -1].reindex(df_prices.index, method="ffill")
        
        # ✅ SPY - FIX YFINANCE COLUMNS
        spy_raw = yf.download("SPY", start=df_prices.index.min(), end=fecha + timedelta(days=1), progress=False, auto_adjust=False)
        if 'Adj Close' in spy_raw.columns:
            spy_prices = spy_raw["Adj Close"].reindex(df_prices.index, method="ffill")
        elif 'Close' in spy_raw.columns:
            spy_prices = spy_raw["Close"].reindex(df_prices.index, method="ffill")
        else:
            spy_prices = spy_raw.iloc[:, -1].reindex(df_prices.index, method="ffill")
        spy_ret = np.log(spy_prices / spy_prices.shift(1)).dropna()
    except:
        # Fallback sin datos macro
        vix = pd.Series(20, index=df_prices.index)  # VIX promedio
        spy_ret = pd.Series(0, index=df_prices.index)

    # Calcular retornos
    df_ret = np.log(df_prices / df_prices.shift(1))

    predicciones = []
    tickers_validos = []

    for ticker in tickers:
        if ticker not in modelos:
            continue
            
        try:
            prices = df_prices[ticker]
            returns = df_ret[ticker]
            
            # === CALCULAR TODAS LAS 23+ FEATURES ===
            
            # Retornos multi-periodo
            ret_1d = returns.loc[fecha]
            ret_5d = returns.rolling(5).sum().loc[fecha]
            ret_10d = returns.rolling(10).sum().loc[fecha]
            ret_20d = returns.rolling(20).sum().loc[fecha]
            
            # Volatilidades
            vol_5d = returns.rolling(5).std().loc[fecha]
            vol_10d = returns.rolling(10).std().loc[fecha]
            vol_20d = returns.rolling(20).std().loc[fecha]
            vol_60d = returns.rolling(60).std().loc[fecha]
            
            # Momentum
            momentum_5d = ret_5d / (vol_5d + 1e-6)
            momentum_20d = ret_20d / (vol_20d + 1e-6)
            rsi_14 = calculate_rsi(prices, 14).loc[fecha] / 100
            
            # Moving averages
            ma_5 = prices.rolling(5).mean().loc[fecha]
            ma_20 = prices.rolling(20).mean().loc[fecha]
            ma_50 = prices.rolling(50).mean().loc[fecha]
            ma_200 = prices.rolling(200).mean().loc[fecha]
            
            ma_ratio_5_20 = (ma_5 / ma_20 - 1) if ma_20 > 0 else 0
            price_to_ma_50 = (prices.loc[fecha] / ma_50 - 1) if ma_50 > 0 else 0
            price_to_ma_200 = (prices.loc[fecha] / ma_200 - 1) if ma_200 > 0 else 0
            
            # Clip ratios
            ma_ratio_5_20 = np.clip(ma_ratio_5_20, -0.5, 0.5)
            price_to_ma_50 = np.clip(price_to_ma_50, -0.5, 0.5)
            price_to_ma_200 = np.clip(price_to_ma_200, -0.5, 0.5)
            
            # Technical indicators
            bollinger_pos = bollinger_position(prices, 20).loc[fecha]
            price_deviation = np.clip((prices.loc[fecha] - ma_20) / ma_20, -0.3, 0.3) if ma_20 > 0 else 0
            
            returns_mean = returns.rolling(60).mean().loc[fecha]
            returns_std = returns.rolling(60).std().loc[fecha]
            returns_z_score = (returns.loc[fecha] - returns_mean) / (returns_std + 1e-8)
            returns_z_score = np.clip(returns_z_score, -3, 3)
            
            # Cross-asset features
            corr_spy_20d = returns.rolling(20).corr(spy_ret.reindex(returns.index)).loc[fecha]
            
            # Beta calculation
            covariance = returns.rolling(60).cov(spy_ret.reindex(returns.index)).loc[fecha]
            spy_variance = spy_ret.reindex(returns.index).rolling(60).var().loc[fecha]
            beta_to_market = (covariance / (spy_variance + 1e-8)) if spy_variance > 1e-8 else 0
            beta_to_market = np.clip(beta_to_market, -3, 3)
            
            # Macro features
            vix_level = np.clip(vix.loc[fecha] / 100, 0, 1) if not pd.isna(vix.loc[fecha]) else 0.2
            vix_change_5d = np.clip(vix.pct_change(5).loc[fecha], -1, 1) if not pd.isna(vix.pct_change(5).loc[fecha]) else 0
            
            vix_ret_series = vix.pct_change()
            corr_vix_20d = returns.rolling(20).corr(vix_ret_series.reindex(returns.index)).loc[fecha]
            
            # Volume proxy features
            vol_ratio_20d = np.clip(vol_5d / (vol_20d + 1e-8), 0, 5)
            vol_spike = float(vol_5d > returns.rolling(20).std().rolling(20).quantile(0.8).loc[fecha])
            
            # === CONSTRUIR VECTOR DE FEATURES ===
            feature_vector = pd.DataFrame({
                'ret_1d': [ret_1d],
                'ret_5d': [ret_5d],
                'ret_10d': [ret_10d],
                'ret_20d': [ret_20d],
                'vol_5d': [vol_5d],
                'vol_10d': [vol_10d],
                'vol_20d': [vol_20d],
                'vol_60d': [vol_60d],
                'momentum_5d': [momentum_5d],
                'momentum_20d': [momentum_20d],
                'rsi_14': [rsi_14],
                'ma_ratio_5_20': [ma_ratio_5_20],
                'price_to_ma_50': [price_to_ma_50],
                'price_to_ma_200': [price_to_ma_200],
                'bollinger_position': [bollinger_pos],
                'price_deviation': [price_deviation],
                'returns_z_score': [returns_z_score],
                'corr_spy_20d': [corr_spy_20d],
                'beta_to_market': [beta_to_market],
                'corr_vix_20d': [corr_vix_20d],
                'vix_level': [vix_level],
                'vix_change_5d': [vix_change_5d],
                'vol_ratio_20d': [vol_ratio_20d],
                'vol_spike': [vol_spike]
            })
            
            # Reemplazar NaN por 0
            feature_vector = feature_vector.fillna(0)
            
            # Predicción
            modelo = modelos[ticker]
            pred = modelo.predict(feature_vector)[0]
            
            predicciones.append(pred)
            tickers_validos.append(ticker)
            
        except Exception as e:
            # Skip este ticker si hay errores
            continue

    return pd.Series(predicciones, index=tickers_validos, name="ret_esperado") 