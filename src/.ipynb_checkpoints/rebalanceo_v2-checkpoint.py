import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from src import config as cfg


class PortfolioProblemV2(Problem):
    def __init__(self, r_hat, Sigma, w_prev, crypto_idx=None, tau=0.4, n_scenarios=2000):
        super().__init__(n_var=len(r_hat), n_obj=3, n_constr=3, xl=0.0, xu=0.25)
        self.r_hat = r_hat
        self.Sigma = (Sigma + Sigma.T) / 2  # reforzar simetría
        self.w_prev = w_prev
        self.tau = tau
        self.crypto_idx = crypto_idx if crypto_idx is not None else []
        self._generate_scenarios(n_scenarios)

    def _generate_scenarios(self, n_scenarios):
        try:
            if self.Sigma is None or np.isnan(self.Sigma).any() or np.isinf(self.Sigma).any():
                raise ValueError("Sigma inválida")

            mean = np.zeros_like(self.r_hat)
            self.scenarios = np.random.multivariate_normal(mean, self.Sigma, size=n_scenarios).T

        except Exception as e:
            print(f"❌ Error generando escenarios en rebalanceo_v2: {e}")
            self.scenarios = None

    def _evaluate(self, X, out, *args, **kwargs):
        if self.scenarios is None:
            raise ValueError("❌ self.scenarios es None: fallo al generar escenarios")

        # Objetivo 1: -retorno esperado
        f1 = -np.dot(X, self.r_hat)

        # Objetivo 2: CVaR95
        losses = -X @ self.scenarios
        var95 = np.percentile(losses, 95, axis=1)
        cvar95 = np.array([l[l >= v].mean() for l, v in zip(losses, var95)])
        f2 = cvar95

        # Objetivo 3: ∑w² → concentración
        f3 = np.sum(X**2, axis=1)

        # Restricción 1: suma de pesos ≈ 1
        g1 = np.abs(X.sum(axis=1) - 1)

        # Restricción 2: peso cripto (opcional)
        g2 = X[:, self.crypto_idx].sum(axis=1) - cfg.CRYPTO_MAX

        # Restricción 3: turnover ≤ τ
        g3 = np.sum(np.abs(X - self.w_prev), axis=1) - self.tau

        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2, g3])


def resolver_optimizacion_v2(r_hat, Sigma, w_prev, tau=0.4, crypto_idx=None):
    problem = PortfolioProblemV2(r_hat, Sigma, w_prev, tau=tau, crypto_idx=crypto_idx)

    algorithm = NSGA2(
        pop_size=cfg.POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", cfg.N_GENS)

    res = minimize(problem, algorithm, termination, seed=42, verbose=False)

    return res


def elegir_w_star_v2(res, r_hat, Sigma, alpha=0.3, beta=0.3, gamma=0.4):
    """
    Escoge la solución más balanceada del frente de Pareto.
    Pondera: retorno esperado, CVaR95 y concentración.
    """
    X = res.X

    # Recalcular CVaR con nuevos escenarios para mayor robustez
    Sigma = (Sigma + Sigma.T) / 2
    scenarios = np.random.multivariate_normal(np.zeros_like(r_hat), Sigma, size=2000).T
    losses = -X @ scenarios
    var95 = np.percentile(losses, 95, axis=1)
    cvar95 = np.array([l[l >= v].mean() for l, v in zip(losses, var95)])

    f1 = -X @ r_hat
    f2 = cvar95
    f3 = np.sum(X**2, axis=1)

    score = alpha * f1 + beta * f2 + gamma * f3
    idx = np.argmin(score)
    return X[idx]
