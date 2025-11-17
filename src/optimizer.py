import pandas as pd
import numpy as np
from typing import Optional
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

ALPHA = 0.95            # Tingkat kepercayaan 95% untuk CVaR

# Calculate CVaR
def calculate_cvar(portfolio_return_series: np.ndarray, alpha: float = ALPHA) -> float:
    losses = -portfolio_return_series
    var_alpha = np.quantile(losses, alpha)
    cvar = losses[losses > var_alpha].mean()

    if np.isnan(cvar):
        return 0.0
    
    return float(cvar)

# Class for Pymoo so it can be solved with NSGA
class PortfolioOptimizationProblem(Problem):
    def __init__(self, mu_vector: np.ndarray, R_matrix: np.ndarray, n_stocks: int, w_max: int):
        
        self.mu_vector = mu_vector
        self.R_matrix = R_matrix
        
        super().__init__(
            n_var=n_stocks,
            n_obj=2,       
            n_constr=1,
            xl=0.0,        
            xu=w_max
        )

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs) -> None:
        
        # ------ Normalize Weights ------ #
        weights = X / X.sum(axis=1)[:, None]
        
        # ------ Calculate Objective function ------ #
        
        # Mean
        f1_mean = weights @ self.mu_vector
        
        # CVaR (Objektif 2) [Persamaan 3.4] [cite: 577-578]
        portfolio_return_series = weights @ self.R_matrix.T
        f2_cvar = np.apply_along_axis(calculate_cvar, 1, portfolio_return_series)
        
        # Pymoo default is minimize
        # Since we wants to maximize, change f1 as negative
        out["F"] = np.column_stack([-f1_mean, f2_cvar])

        # ----- Calculate Constraints ----- #
        # g(x) <= 0
        g1 = np.abs(weights.sum(axis=1) - 1.0)
        out["G"] = np.column_stack([g1])

def run_optimization(r_matrix: pd.DataFrame, mu_vector: pd.DataFrame, n_population: int, n_generations: int, n_stocks: int, w_max: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    print('----- Optimisasi dengan NSGA II -----')
    print("Memulai optimisasi...")

    # Initialize
    portofolio = PortfolioOptimizationProblem(mu_vector, r_matrix, n_stocks, w_max)
    nsga2 = NSGA2(
        pop_size=n_population,
    )
    
    # Run
    print(f"\nMenjalankan NSGA-II...")
    print(f"   Populasi: {n_population}")
    print(f"   Generasi: {n_generations}")
    res = minimize(
        problem=portofolio,
        algorithm=nsga2,
        termination=('n_gen', n_generations),
        seed=1,
        verbose=True
    )
    
    print("\nOptimisasi Selesai!")
    
    # Evaluate and prepare for visualization
    if res.F is not None:
        print(f"\nBerhasil menemukan {len(res.F)} solusi non-dominan (Pareto Front).")
        
        pareto_front_F = res.F
        pareto_weights_X = res.X
        
        # Convert back F1 mean to positive
        pareto_front_F[:, 0] = -pareto_front_F[:, 0]
        
        return pareto_front_F, pareto_weights_X
        
    else:
        print("\nOptimisasi gagal menemukan solusi.")
        return None
