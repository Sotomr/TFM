# ─── src/config.py ─────────────────────────────────────────────
# Parámetros globales (versión inicial – iremos ampliando luego)
from pathlib import Path
WINDOW        = 60      # tamaño de la ventana look-back (días)
REBAL_FREQ    = 10      # frecuencia de rebalanceo (días)
TARGET_HORIZON= 1       # horizonte de predicción actual (1 día)
MODEL_LIST    = ["lstm_t1"]   # lista de modelos que el back-test va a usar
ALGO          = "nsga2"       # algoritmo evolutivo por defecto
W_MAX         = 0.20          # peso máximo por activo
CRYPTO_MAX    = 0.10          # techo combinado BTC+ETH
COST_TRADE    = 0.002        # coste total por rotación (0.2 %)
MAX_TURNOVER = 0.5
RANDOM_SEED = 42


ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
MODELS = ROOT / "models"
RESULT = ROOT / "results"

MODEL_TYPE = "cnn5d"           # o "xgb", para controlar cuál se usa
LSTM_MODEL_NAME  = "lstm_t1.keras"
LSTM5D_MODEL_NAME = "lstm5d.keras"
GRU5D_MODEL_NAME = "gru5d.keras"
XGB_MODEL_NAME = "xgb5d"
CNN5D_MODEL_NAME    = "cnn5d.keras"


POP_SIZE = 300    
N_GENS   = 450    
TAU            = 0.4         # turnover máximo
CRYPTO_IDX     = [0,1]       # índices de BTC, ETH en tu vector de activos


START_BACKTEST = "2023-07-01"


