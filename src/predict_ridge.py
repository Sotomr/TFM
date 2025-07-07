import pandas as pd
import numpy as np
import joblib
from src import config as cfg


def predict_ridge(df_prices: pd.DataFrame, fecha: pd.Timestamp, tickers: list[str]) -> pd.Series:
    """Predice retornos esperados a 5 d\u00edas usando modelos Ridge por ticker."""
    modelos = joblib.load(cfg.MODELS / "ridge.pkl")

    if fecha not in df_prices.index:
        raise ValueError(f"La fecha {fecha} no est\u00e1 en el \u00edndice del dataframe")
    idx = df_prices.index.get_loc(fecha)
    if idx < 6:
        raise ValueError("No hay suficientes d\u00edas anteriores para calcular features")

    df_ret = np.log(df_prices / df_prices.shift(1))
    df_ret5 = df_ret.rolling(5).sum()
    df_vol5 = df_ret.rolling(5).std()
    df_mom = (df_ret5 / (df_vol5 + 1e-6)).clip(-10, 10)

    datos = []
    for ticker in tickers:
        try:
            datos.append(
                (
                    ticker,
                    [
                        df_ret.loc[fecha, ticker],
                        df_ret5.loc[fecha, ticker],
                        df_vol5.loc[fecha, ticker],
                        df_mom.loc[fecha, ticker],
                    ],
                )
            )
        except KeyError:
            continue

    if not datos:
        return pd.Series(dtype=float)

    X = pd.DataFrame([x[1] for x in datos], columns=["ret_1d", "ret_5d", "vol_5d", "momentum"])
    tickers_ok = [x[0] for x in datos]

    preds = []
    for i, ticker in enumerate(tickers_ok):
        model = modelos.get(ticker)
        if model is not None:
            pred = model.predict(X.iloc[[i]])[0]
        else:
            pred = np.nan
        preds.append(pred)

    return pd.Series(preds, index=tickers_ok, name="ret_esperado")
