import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from src import config as cfg

def predict_xgb(df_prices: pd.DataFrame, fecha: pd.Timestamp, tickers: list[str]) -> pd.Series:
    """
    Predice retornos esperados a 5 días usando modelos LightGBM por ticker.
    
    Parámetros:
    - df_prices: DataFrame con precios (índice: fechas, columnas: tickers)
    - fecha: fecha en la que se quiere predecir (usa t-1 para features)
    - tickers: lista de tickers a procesar

    Devuelve:
    - Serie con retornos esperados a 5 días, indexada por ticker
    """
    # Cargar modelos
    modelos = joblib.load(cfg.DATA / "processed" / "xgb_model.pkl")
    
    # Asegurarse de que hay suficientes datos
    if fecha not in df_prices.index:
        raise ValueError(f"La fecha {fecha} no está en el índice del dataframe")
    
    idx = df_prices.index.get_loc(fecha)
    if idx < 6:
        raise ValueError("No hay suficientes días anteriores para calcular features")

    df_ret = np.log(df_prices / df_prices.shift(1))
    df_ret5 = df_ret.rolling(5).sum()
    df_vol5 = df_ret.rolling(5).std()
    df_mom = (df_ret5 / (df_vol5 + 1e-6)).clip(-10, 10)

    datos = []
    for ticker in tickers:
        try:
            x = {
                "ret_1d": df_ret.loc[fecha, ticker],
                "ret_5d": df_ret5.loc[fecha, ticker],
                "vol_5d": df_vol5.loc[fecha, ticker],
                "momentum": df_mom.loc[fecha, ticker],
            }
            datos.append((ticker, list(x.values())))
        except KeyError:
            continue  # ticker no disponible en esa fecha

    df_pred = pd.DataFrame([x[1] for x in datos], columns=["ret_1d", "ret_5d", "vol_5d", "momentum"])
    tickers_ok = [x[0] for x in datos]

    predicciones = []
    for i, ticker in enumerate(tickers_ok):
        modelo = modelos.get(ticker)
        if modelo is not None:
            x_input = pd.DataFrame([df_pred.iloc[i]], columns=["ret_1d", "ret_5d", "vol_5d", "momentum"])
            pred = modelo.predict(x_input)[0]
        else:
            pred = np.nan
        predicciones.append(pred)

    return pd.Series(predicciones, index=tickers_ok, name="ret_esperado")
