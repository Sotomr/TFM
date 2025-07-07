# === evol_utils.py ===
import numpy as np
from src import config as cfg
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


# En config.py define, si aún no lo tienes:
# cfg.MAX_TURNOVER = 0.40     # 40 % máx. de rotación por rebalanceo

TOL = 1e-4                    # holgura |∑w−1| ≤ TOL


# ───────────────────────────────── helpers ──────────────────────────────────
def _risk(mat_w: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Varianza vectorizada que sirve tanto para w shape (n,) como (m, n)."""
    return np.sum((mat_w @ sigma) * mat_w, axis=-1)


# ────────────────────────── optimizador NSGA-II ─────────────────────────────
def resolver_optimizacion(mu_hat: np.ndarray,
                           Sigma: np.ndarray,
                           w_prev: np.ndarray | None = None):
    """
    Devuelve la frontera de Pareto (riesgo, retorno neto) con restricción dura
    de rotación ∑|Δw| ≤ τ.
    """
    n = len(mu_hat)
    btc_idx, eth_idx = 0, 1          # ajusta si cambia el orden de columnas

    class PortOpt(Problem):
        def __init__(self, w_prev):
            super().__init__(n_var=n, n_obj=2, n_constr=3, xl=0.0, xu=cfg.W_MAX)
            self.w_prev = w_prev

        def _evaluate(self, X, out, *args, **kwargs):
            X = np.atleast_2d(X)                 # garantiza 2-D siempre
            risk = _risk(X, Sigma)

            bruto = X @ mu_hat
            if self.w_prev is not None:
                turnover = np.abs(X - self.w_prev).sum(axis=1)
                neto = bruto - turnover * cfg.COST_TRADE
                g_turn = turnover - cfg.MAX_TURNOVER     # ≤ 0  ⇒ rotación ≤ τ
            else:
                turnover = np.zeros(X.shape[0])
                neto = bruto
                g_turn = np.zeros_like(risk)             # sin límite en t₀

            ret = -neto                                  # minimizar

            # ---------- restricciones ----------
            g1 = np.abs(X.sum(axis=1) - 1.0) - TOL       # |∑w−1| ≤ TOL
            g2 = X[:, btc_idx] + X[:, eth_idx] - cfg.CRYPTO_MAX

            out["F"] = np.column_stack([risk, ret])
            out["G"] = np.column_stack([g1, g2, g_turn])


    np.random.seed(cfg.RANDOM_SEED)

    res = minimize(
        PortOpt(w_prev),
        NSGA2(pop_size=cfg.POP_SIZE),
        termination=('n_gen', cfg.N_GENS),
        verbose=False
    )
    return res


# ───────────────────────────── selector final ───────────────────────────────
def elegir_w_star(res,
                  mu_hat: np.ndarray,
                  Sigma: np.ndarray,
                  w_prev: np.ndarray | None = None) -> np.ndarray:
    """Elige la cartera con mejor Sharpe basado en retorno neto."""
    if res.X is None:
        raise RuntimeError("NSGA-II no encontró soluciones factibles.")

    W = np.atleast_2d(res.X)
    risks = _risk(W, Sigma)
    bruto = W @ mu_hat

    if w_prev is not None:
        turns = np.abs(W - w_prev).sum(axis=1)
        neto = bruto - turns * cfg.COST_TRADE
    else:
        neto = bruto

    sharpe_neto = neto / np.sqrt(risks + 1e-8)
    return W[np.argmax(sharpe_neto)]
