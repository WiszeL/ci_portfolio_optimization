import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

ALPHA = 0.95


def calculate_cvar(portfolio_return_series: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Calculates Conditional Value at Risk (CVaR) for a series of portfolio returns.
    """
    losses = -portfolio_return_series
    var_alpha = np.quantile(losses, alpha)
    tail_losses = losses[losses > var_alpha]

    if tail_losses.size == 0:
        return 0.0

    cvar = tail_losses.mean()

    if np.isnan(cvar):
        return 0.0

    return float(cvar)


class PortfolioOptimizationProblem(Problem):
    def __init__(
        self, mu_vector: np.ndarray, R_matrix: np.ndarray, n_stocks: int, w_max: float
    ):
        """
        Strategy: Mean-CVaR Optimization with Repair Constraint handling.
        """
        self.mu_vector = mu_vector
        self.R_matrix = R_matrix
        self.w_max = w_max

        # n_constr=1 because "Sum of weights = 1" is handled via Repair (Normalization)
        super().__init__(n_var=n_stocks, n_obj=2, n_constr=1, xl=0.0, xu=1.0)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        # Constraint Handling 1: Soft Repair for Budget Constraint (Sum = 1)
        weights = X / (X.sum(axis=1)[:, None] + 1e-8)

        # Objective 1: Mean Return (Maximization -> Minimization)
        f1_mean = weights @ self.mu_vector

        # Objective 2: CVaR (Minimization)
        portfolio_return_series = weights @ self.R_matrix.T
        f2_cvar = np.apply_along_axis(calculate_cvar, 1, portfolio_return_series)

        out["F"] = np.column_stack([-f1_mean, f2_cvar])

        # Constraint Handling 2: Max Weight Constraint
        g2 = (weights - self.w_max).max(axis=1)
        out["G"] = np.column_stack([g2])


def run_optimization(
    r_matrix: pd.DataFrame,
    mu_vector: pd.Series,
    n_population: int,
    n_generations: int,
    n_stocks: int,
    w_max: float,
    algorithm: Algorithm | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:

    # Data Type Conversion
    mu_numpy = mu_vector.to_numpy().flatten()
    r_numpy = r_matrix.to_numpy()

    # Initialize Problem
    portofolio = PortfolioOptimizationProblem(mu_numpy, r_numpy, n_stocks, w_max)

    # Algorithm Configuration (Dependency Injection)
    # Jika user tidak kasih algoritma, pakai default NSGA-II (Sesuai Proposal)
    if algorithm is None:
        algo_name = "NSGA-II (Default)"
        algorithm_instance = NSGA2(pop_size=n_population)
    else:
        # Jika user kasih (misal NSGA-III), pakai itu
        # Kita override pop_size biar adil sesuai input parameter
        algo_name = algorithm.__class__.__name__

        # Pylance tidak tahu kalau subclass (NSGA3) punya pop_size, jadi kita set dinamis.
        if hasattr(algorithm, "pop_size"):
            setattr(algorithm, "pop_size", n_population)

        algorithm_instance = algorithm

    print(f"----- Optimisasi dengan {algo_name} -----")
    print(f"   Populasi: {n_population}")
    print(f"   Generasi: {n_generations}")

    # Execute
    res = minimize(
        problem=portofolio,
        algorithm=algorithm_instance,
        termination=("n_gen", n_generations),
        seed=1,
        verbose=True,
    )

    print("\nOptimisasi Selesai!")

    if res.F is not None and res.X is not None:
        print(f"Berhasil menemukan {len(res.F)} solusi non-dominan.")

        pareto_front_F = res.F
        pareto_raw_X = res.X

        # Final Normalization
        pareto_weights_X = pareto_raw_X / (pareto_raw_X.sum(axis=1)[:, None] + 1e-8)

        # Convert Mean Return back to positive
        pareto_front_F[:, 0] = -pareto_front_F[:, 0]

        return pareto_front_F, pareto_weights_X

    else:
        print("Optimisasi gagal.")
        return None
