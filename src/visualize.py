import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def print_portfolio_analysis(title: str, tickers: List[str], weights: np.ndarray, 
                             mean_return: float, cvar_risk: float):   
    print("\n" + "="*50)
    print(title)
    print("="*50)
    print(f"  > Mean Return (f1): {mean_return:.6f} (Daily)")
    print(f"  > CVaR Risk (f2)  : {cvar_risk:.6f} (Daily @ 95%)")
    print("\n  Komposisi Bobot (di atas 0.5%):")
    
    # Combine tickers and weights, sort descending
    portfolio = sorted(
        [(ticker, weight) for ticker, weight in zip(tickers, weights)],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print only significant weights
    for ticker, weight in portfolio:
        if weight > 0.005: # Only display if weight > 0.5%
            print(f"    - {ticker:<8}: {weight:>6.2%}")
    print("="*50)

def visualize_results(
    pareto_front_F: np.ndarray, 
    pareto_weights_X: np.ndarray, 
    tickers: List[str]
):
    print("Memulai visualisasi dan analisis...")
        
    # Quick validation
    if len(tickers) != pareto_weights_X.shape[1]:
        print(f"Error: Mismatch! Tickers ({len(tickers)}) vs Weights ({pareto_weights_X.shape[1]})")
        return

    # Extract data for plotting
    # Col 0 = Mean_Return, Col 1 = CVaR_Risk
    mean_returns = pareto_front_F[:, 0]
    cvar_risks = pareto_front_F[:, 1]
    all_weights = pareto_weights_X
    
    # --- Analyze 3 Key Solution Points ---
    
    # 1. Minimum Risk Portfolio (Leftmost point)
    min_risk_idx = np.argmin(cvar_risks)
    min_risk_ret = mean_returns[min_risk_idx]
    min_risk_cvar = cvar_risks[min_risk_idx]
    min_risk_weights = all_weights[min_risk_idx]
    
    # 2. Maximum Return Portfolio (Topmost point)
    max_ret_idx = np.argmax(mean_returns)
    max_ret_ret = mean_returns[max_ret_idx]
    max_ret_cvar = cvar_risks[max_ret_idx]
    max_ret_weights = all_weights[max_ret_idx]
    
    # 3. Balanced (Knee) Portfolio
    #    (Found using normalization & euclidean distance)
    norm_ret = (mean_returns - min_risk_ret) / (max_ret_ret - min_risk_ret)
    norm_cvar = (cvar_risks - min_risk_cvar) / (max_ret_cvar - min_risk_cvar)
    distances = np.sqrt((norm_ret - 1)**2 + (norm_cvar - 0)**2)
    knee_idx = np.argmin(distances)
    knee_ret = mean_returns[knee_idx]
    knee_cvar = cvar_risks[knee_idx]
    knee_weights = all_weights[knee_idx]
    
    # --- Plot Pareto Front ---
    
    plt.figure(figsize=(12, 8))
    
    # Plot all solutions
    plt.scatter(cvar_risks, mean_returns, c='blue', 
                s=15, alpha=0.5, label=f'Pareto Optimal Solutions (n={len(mean_returns)})')
    
    # Highlight the 3 key points
    plt.scatter(min_risk_cvar, min_risk_ret, c='green', marker='*', 
                s=200, label=f'1. Min Risk (CVaR={min_risk_cvar:.5f})', zorder=10)
    
    plt.scatter(max_ret_cvar, max_ret_ret, c='red', marker='*', 
                s=200, label=f'2. Max Return (Mean={max_ret_ret:.5f})', zorder=10)
    
    plt.scatter(knee_cvar, knee_ret, c='orange', marker='X', 
                s=150, label=f'3. Balanced/Knee Point', zorder=10)
    
    # Plot configuration
    plt.title('Hasil Optimisasi NSGA-II: Pareto Optimal Front (Mean vs. CVaR)', fontsize=16)
    plt.xlabel('Risiko (CVaR @ 95%) - Objektif f2 (Minimasi)', fontsize=12)
    plt.ylabel('Return (Mean Harian) - Objektif f1 (Maksimasi)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Format axes to percentage for readability
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3%}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3%}'))
    
    print("\nVisualisasi Pareto Front berhasil dibuat.")
    plt.savefig("result/pareto_front_plot.png")
    print("Plot disimpan sebagai 'pareto_front_plot.png'")
    plt.show() # Display plot in notebook
    
    # --- Print Detailed Analysis of Key Points ---
    
    print_portfolio_analysis(
        "1. Portofolio Risiko Minimum (Paling Konservatif)",
        tickers, min_risk_weights, min_risk_ret, min_risk_cvar
    )
    print_portfolio_analysis(
        "3. Portofolio Titik Tengah (Balanced/Knee)",
        tickers, knee_weights, knee_ret, knee_cvar
    )
    print_portfolio_analysis(
        "2. Portofolio Return Maksimum (Paling Agresif)",
        tickers, max_ret_weights, max_ret_ret, max_ret_cvar
    )