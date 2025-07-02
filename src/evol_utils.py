# === evol_utils.py ===
import numpy as np
from src import config as cfg
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


# cfg.MAX_TURNOVER = 0.40   # ≤ 40 % de la cartera por rebalanceo (defínelo en config)

TOL = 1e-4                  # holgura numérica para ∑w = 1


def _risk(mat_w, sigma):
    """Varianza vectorizada que funciona con shape (n,) o (m,n)."""
    return np.sum((mat_w @ sigma) * mat_w, axis=-1)


def resolver_optimizacion(mu_hat: np.ndarray,
                           Sigma: np.ndarray,
                           w_prev: np.ndarray | None = None):
    n = len(mu_hat)
    btc_idx, eth_idx = 0, 1     # ajusta si cambia el orden de columnas

    class PortOpt(Problem):
        def __init__(self, w_prev):
            super().__init__(n_var=n, n_obj=2, n_constr=4, xl=0.0, xu=cfg.W_MAX)
            self.w_prev = w_prev

        def _evaluate(self, X, out, *args, **kwargs):
            X = np.atleast_2d(X)                       # garantiza 2-D
            risk = _risk(X, Sigma)

            bruto = X @ mu_hat
            if self.w_prev is not None:
                turnover = np.sum(np.abs(X - self.w_prev), axis=1)
                neto = bruto - turnover * cfg.COST_TRADE
            else:
                turnover = np.zeros(X.shape[0])
                neto = bruto
            ret = -neto                                # minimizar

            # ---------- restricciones ----------
            suma = X.sum(axis=1)
            g1_pos = suma - 1 + TOL                   #  ≤ 0  →  suma ≤ 1+tol
            g1_neg = -suma + 1 + TOL                  #  ≤ 0  →  suma ≥ 1-tol
            g2 = X[:, btc_idx] + X[:, eth_idx] - cfg.CRYPTO_MAX
            g_turn = (turnover - cfg.MAX_TURNOVER
                      if self.w_prev is not None
                      else np.zeros_like(risk))

            out["F"] = np.column_stack([risk, ret])
            out["G"] = np.column_stack([g1_pos, g1_neg, g2, g_turn])

    res = minimize(
        PortOpt(w_prev),
        NSGA2(pop_size=cfg.POP_SIZE),
        termination=('n_gen', cfg.N_GENS),
        verbose=False
    )
    return res


def elegir_w_star(res,
                  mu_hat: np.ndarray,
                  Sigma: np.ndarray,
                  w_prev: np.ndarray | None = None) -> np.ndarray:
    """Devuelve la cartera con mejor Sharpe sobre retorno neto."""
    if res.X is None:
        raise RuntimeError("NSGA-II no encontró soluciones factibles "
                           "(ajusta τ o el tol de la suma).")

    W = np.atleast_2d(res.X)
    risks = _risk(W, Sigma)
    bruto = W @ mu_hat

    if w_prev is not None:
        turns = np.sum(np.abs(W - w_prev), axis=1)
        neto = bruto - turns * cfg.COST_TRADE
    else:
        neto = bruto

    sharpe = neto / np.sqrt(risks + 1e-8)
    return W[np.argmax(sharpe)]
