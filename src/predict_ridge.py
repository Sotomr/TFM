import pandas as pd
import numpy as np
import joblib
from src import config as cfg


def predict_ridge(df_prices: pd.DataFrame, fecha: pd.Timestamp, tickers: list[str]) -> pd.Series:
    """Predice retornos esperados a 5 días usando modelos Ridge por ticker."""
    modelos = joblib.load(cfg.MODELS / "ridge.pkl")

    if fecha not in df_prices.index:
        raise ValueError(f"La fecha {fecha} no está en el índice del dataframe")
    idx = df_prices.index.get_loc(fecha)
    if idx < 6:
        raise ValueError("No hay suficientes días anteriores para calcular features")

    # ✅ CALCULAR FEATURES CON MANEJO DE NaN
    df_ret = np.log(df_prices / df_prices.shift(1))
    df_ret5 = df_ret.rolling(5).sum()
    df_vol5 = df_ret.rolling(5).std()
    df_mom = (df_ret5 / (df_vol5 + 1e-6)).clip(-10, 10)

    datos = []
    for ticker in tickers:
        try:
            # Extraer features para este ticker en esta fecha
            features = [
                df_ret.loc[fecha, ticker],
                df_ret5.loc[fecha, ticker],
                df_vol5.loc[fecha, ticker],
                df_mom.loc[fecha, ticker],
            ]

            if any(pd.isna(val) or np.isinf(val) for val in features):
                # Reemplazar con valores seguros si hay NaN/Inf
                features = [
                    0.0 if pd.isna(features[0]) or np.isinf(features[0]) else features[0],  # ret_1d
                    0.0 if pd.isna(features[1]) or np.isinf(features[1]) else features[1],  # ret_5d  
                    0.01 if pd.isna(features[2]) or np.isinf(features[2]) else features[2], # vol_5d
                    0.0 if pd.isna(features[3]) or np.isinf(features[3]) else features[3],  # momentum
                ]
            
            datos.append((ticker, features))
            
        except KeyError:
            continue

    if not datos:
        return pd.Series(dtype=float)

    X = pd.DataFrame([x[1] for x in datos], columns=["ret_1d", "ret_5d", "vol_5d", "momentum"])
    tickers_ok = [x[0] for x in datos]
    
    X = X.fillna(0)
    
    X['ret_1d'] = X['ret_1d'].clip(-0.2, 0.2)
    X['ret_5d'] = X['ret_5d'].clip(-0.5, 0.5)
    X['vol_5d'] = X['vol_5d'].clip(0.001, 1.0)  # Volatilidad positiva
    X['momentum'] = X['momentum'].clip(-10, 10)

    preds = []
    for i, ticker in enumerate(tickers_ok):
        model = modelos.get(ticker)
        if model is not None:
            # ✅ VERIFICACIÓN ANTES DE PREDICCIÓN
            X_row = X.iloc[[i]]
            if X_row.isna().any().any():
                print(f"⚠️ Warning: NaN detectado en features para {ticker}")
                pred = 0.0  # Predicción segura
            else:
                pred = model.predict(X_row)[0]
        else:
            pred = 0.0  # Predicción segura si no hay modelo
        preds.append(pred)

    return pd.Series(preds, index=tickers_ok, name="ret_esperado")
