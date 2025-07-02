import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Clase de problema (como ya usaste)
class PortfolioProblem(Problem):
    def __init__(self, r_hat, Sigma):
        super().__init__(n_var=len(r_hat), n_obj=3, n_constr=2, xl=0.0, xu=0.2)
        self.r_hat = r_hat
        self.Sigma = Sigma

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = -np.dot(X, self.r_hat)
        f2 = np.einsum("ij,ij->i", X @ self.Sigma, X)
        f3 = np.sum(X**2, axis=1)
        tol = 1e-4
        g1 = np.abs(X.sum(axis=1) - 1) - tol   # ≤ 0  supone “suma ~ 1”

        g2 = X[:, -2] + X[:, -1] - 0.1
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])
