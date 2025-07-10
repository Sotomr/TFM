import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from scipy.stats import norm
# ✅ AÑADIDO para reproducibilidad
from src import config as cfg

def scen_simul(r_hat, Sigma, n_scen=5_000):
    """Muestra escenarios P&L ~ N(r_hat, Σ).  Shape -> (n_scen, N)"""
    # ✅ CRÍTICO: Fijar semilla para reproducibilidad
    np.random.seed(cfg.RANDOM_SEED)
    
    L = np.linalg.cholesky(Sigma + 1e-12*np.eye(len(r_hat)))
    z = np.random.randn(n_scen, len(r_hat))
    return r_hat + z @ L.T          # log-returns por activo

def cvar95(pnl):
    """CVaR al 95 % de una serie de P&L de cartera."""
    var = np.percentile(pnl, 5)     # VaR (pérdida negativa)
    return -pnl[pnl <= var].mean()  # CVaR>0: cuanto menor mejor

class PortfolioProblemV2(Problem):
    def __init__(self, R, w_prev, xl=0.0, xu=0.2, tau=0.4):
        """
        R : escenarios de retornos  (S, N)
        w_prev : pesos vigentes en la cartera (N,)
        """
        self.R = R
        self.w_prev = w_prev
        self.tau = tau
        super().__init__(n_var=R.shape[1], n_obj=3, n_constr=3,
                         xl=xl, xu=xu)

    # --- Función de evaluación -----------------------------
    def _evaluate(self, X, out, *args, **kw):
        # X shape -> (pop, N)
        pnl = X @ self.R.T                        # (pop, S)
        mu = pnl.mean(axis=1)
        cvar = np.apply_along_axis(cvar95, 1, pnl)
        g_budget = np.abs(X.sum(axis=1) - 1)     # =0 cumple
        g_turn   = (np.abs(X - self.w_prev).sum(axis=1)
                    - self.tau)                  # ≤0 cumple
        g_crypto = X[:, -2] + X[:, -1] - 0.1     # ejemplo
        out["F"] = np.column_stack([-mu, cvar, (X**2).sum(axis=1)])
        out["G"] = np.column_stack([g_budget, g_turn, g_crypto])

# --- helpers ------------------------------------------------
def resolver_optimizacion_v2(r_hat, Sigma, w_prev, tau=0.4):
    # ✅ REPRODUCIBILIDAD: Fijar semilla antes de simular
    np.random.seed(cfg.RANDOM_SEED)
    
    R = scen_simul(r_hat, Sigma, n_scen=5_000)
    problem = PortfolioProblemV2(R, w_prev, tau=tau)
    algo = NSGA2(pop_size=300,
                 sampling=FloatRandomSampling(),
                 crossover=SBX(prob=0.9, eta=15),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True)
    res = minimize(problem, algo,
                   termination=get_termination("n_gen", 250),
                   seed=cfg.RANDOM_SEED,  # ✅ AÑADIDO
                   verbose=False)
    return res

def elegir_w_star_v2(res):
    # máximo Sharpe usando la media y el desvío de cada solución
    mu  = -res.F[:, 0]
    sig = res.F[:, 1]          # ≈ CVaR, no SD; usar surrogate
    sharpe = mu/(sig + 1e-8)
    return res.X[ sharpe.argmax() ]
