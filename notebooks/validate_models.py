#!/usr/bin/env python3
"""
Script de validación para verificar que todos los modelos del TFM 
generan predicciones únicas y realistas.
"""

import sys, pathlib
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

# Setup
PROJECT_ROOT = pathlib.Path().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config as cfg
from tensorflow import keras

def load_test_data():
    """Cargar datos de test comunes para todos los modelos"""
    prices_path = cfg.DATA / "raw" / "prices.parquet"
    df_prices = pd.read_parquet(prices_path).sort_index()
    
    # Usar mismos tickers que en backtest
    lstm_data = joblib.load(cfg.DATA / "processed" / "lstm_data.pkl")
    tickers = lstm_data["tickers"]
    df_prices = df_prices[tickers]
    df_ret = np.log(df_prices / df_prices.shift(1)).dropna()
    
    # Features: momentum
    ret5 = df_ret.rolling(5).sum()
    vol5 = df_ret.rolling(5).std()
    momentum = (ret5 / vol5).shift(1)
    df_feat = pd.concat([df_ret.shift(1), momentum], axis=1).dropna()
    
    # Fechas de test (últimos 100 días)
    test_dates = df_feat.index[-100:]
    
    return df_feat, df_ret, tickers, test_dates

def validate_model_predictions():
    """Validar que cada modelo produce predicciones únicas"""
    
    print("🔍 VALIDACIÓN DE MODELOS TFM")
    print("=" * 50)
    
    df_feat, df_ret, tickers, test_dates = load_test_data()
    
    models_to_test = {
        "lstm": cfg.MODELS / "lstm_t1.keras",
        "lstm5d": cfg.MODELS / "lstm5d.keras", 
        "cnn5d": cfg.MODELS / "cnn5d.keras",
        "gru5d": cfg.MODELS / "gru5d.keras",
    }
    
    predictions = {}
    
    for model_name, model_path in models_to_test.items():
        if not model_path.exists():
            print(f" {model_name}: Modelo no encontrado en {model_path}")
            continue
            
        print(f"\n Validando {model_name.upper()}...")
        
        try:
            # Cargar modelo y escaladores
            model = keras.models.load_model(model_path, compile=False)
            scaler_X = joblib.load(cfg.MODELS / f"scaler_X_{model_name}.pkl")
            scaler_y = joblib.load(cfg.MODELS / f"scaler_y_{model_name}.pkl")
            
            model_preds = []
            
            # Hacer predicciones en fechas de test
            for fecha in test_dates[-10:]:  # Solo últimas 10 fechas
                try:
                    idx = df_feat.index.get_loc(fecha)
                    if idx < cfg.WINDOW:
                        continue
                        
                    if model_name == "cnn5d":
                        ventana = df_feat.iloc[idx - cfg.WINDOW: idx]
                        X_scaled = scaler_X.transform(ventana.values)
                        n_assets, n_chan = len(tickers), 2
                        X_input = X_scaled.reshape(1, cfg.WINDOW, n_assets, n_chan)
                    else:
                        ventana = df_feat.iloc[idx - cfg.WINDOW: idx]
                        X_scaled = scaler_X.transform(ventana.values)
                        X_input = X_scaled.reshape(1, cfg.WINDOW, -1)
                    
                    # Predicción
                    r_hat = model.predict(X_input, verbose=0)[0]
                    r_hat = scaler_y.inverse_transform([r_hat])[0]
                    
                    # Conversión para modelos 5d
                    if model_name in ["lstm5d", "cnn5d", "gru5d"]:
                        r_hat = r_hat / 5.0
                    
                    r_hat = np.clip(r_hat, -0.12, 0.12)
                    model_preds.append(r_hat)
                    
                except Exception as e:
                    print(f"   Error en fecha {fecha}: {e}")
                    continue
            
            if model_preds:
                pred_array = np.array(model_preds)
                predictions[model_name] = pred_array
                
                # Estadísticas
                print(f"   Predicciones: {len(model_preds)}")
                print(f"   Media: {pred_array.mean():.4f}")
                print(f"   Std:  {pred_array.std():.4f}")
                print(f"   Min:  {pred_array.min():.4f}")
                print(f"   Max:  {pred_array.max():.4f}")
                
        except Exception as e:
            print(f"   ERROR cargando {model_name}: {e}")
    
    # Comparar predicciones entre modelos
    print(f"\n ANÁLISIS DE DIFERENCIAS ENTRE MODELOS")
    print("=" * 50)
    
    if len(predictions) >= 2:
        model_names = list(predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                pred1 = predictions[model1].flatten()
                pred2 = predictions[model2].flatten()
                
                min_len = min(len(pred1), len(pred2))
                pred1, pred2 = pred1[:min_len], pred2[:min_len]
                
                if len(pred1) > 0:
                    corr = np.corrcoef(pred1, pred2)[0,1]
                    rmse = np.sqrt(((pred1 - pred2)**2).mean())
                    
                    print(f"  {model1} vs {model2}:")
                    print(f"    Correlación: {corr:.3f}")
                    print(f"    RMSE:        {rmse:.4f}")
                    
                    if abs(corr) > 0.99:
                        print(f"     ALERTA: Modelos muy similares!")
                    elif abs(corr) < 0.1:
                        print(f"     Buenos: Predicciones diferentes")
                    else:
                        print(f"      Moderado: Correlación media")
    
    print(f"\n VALIDACIÓN COMPLETADA")
    return predictions

if __name__ == "__main__":
    predictions = validate_model_predictions() 
