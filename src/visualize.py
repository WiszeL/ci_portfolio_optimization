import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def print_portfolio_analysis(
    title: str,
    tickers: list[str],
    weights: np.ndarray,
    mean_return: float,
    cvar_risk: float,
) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"  > Mean Return (f1): {mean_return:.6f} (Daily)")
    print(f"  > CVaR Risk (f2)  : {cvar_risk:.6f} (Daily @ 95%)")
    print("\n  Komposisi Bobot (di atas 0.5%):")

    # Sort weights in descending order for better readability
    portfolio = sorted(
        [(ticker, weight) for ticker, weight in zip(tickers, weights)],
        key=lambda x: x[1],
        reverse=True,
    )

    for ticker, weight in portfolio:
        # Filter out negligible weights (< 0.5%) to keep the report clean
        if weight > 0.005:
            print(f"    - {ticker:<8}: {weight:>6.2%}")
    print("=" * 50)


def visualize_results(
    pareto_front_F: np.ndarray, pareto_weights_X: np.ndarray, tickers: list[str]
) -> None:
    print("Memulai visualisasi dan analisis...")

    # Safety check for dimensions
    if len(tickers) != pareto_weights_X.shape[1]:
        print(
            f"Error: Mismatch! Tickers ({len(tickers)}) vs Weights ({pareto_weights_X.shape[1]})"
        )
        return

    # Extract objectives
    mean_returns = pareto_front_F[:, 0]
    cvar_risks = pareto_front_F[:, 1]
    all_weights = pareto_weights_X

    # ---------------------------------------------------------
    # 1. Minimum Risk Portfolio (The "Anchor" on the left)
    # ---------------------------------------------------------
    min_risk_idx = int(np.argmin(cvar_risks))
    min_risk_ret = mean_returns[min_risk_idx]
    min_risk_cvar = cvar_risks[min_risk_idx]
    min_risk_weights = all_weights[min_risk_idx]

    # ---------------------------------------------------------
    # 2. Maximum Return Portfolio (The "Anchor" on the top)
    # ---------------------------------------------------------
    max_ret_idx = int(np.argmax(mean_returns))
    max_ret_ret = mean_returns[max_ret_idx]
    max_ret_cvar = cvar_risks[max_ret_idx]
    max_ret_weights = all_weights[max_ret_idx]

    # ---------------------------------------------------------
    # 3. Balanced / Knee Portfolio (The Best Trade-off)
    # Strategy: Euclidean Distance to "Utopia Point"
    # We normalize both axes to 0-1 range to ensure fair distance calculation.
    # Utopia Point in normalized space: (Risk=0, Return=1)
    # ---------------------------------------------------------
    norm_ret = (mean_returns - min_risk_ret) / (max_ret_ret - min_risk_ret)
    norm_cvar = (cvar_risks - min_risk_cvar) / (max_ret_cvar - min_risk_cvar)

    # Calculate distance to (Return=1, Risk=0)
    distances = np.sqrt((norm_ret - 1) ** 2 + (norm_cvar - 0) ** 2)

    knee_idx = int(np.argmin(distances))
    knee_ret = mean_returns[knee_idx]
    knee_cvar = cvar_risks[knee_idx]
    knee_weights = all_weights[knee_idx]

    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))

    # Plot all Pareto points
    plt.scatter(
        cvar_risks,
        mean_returns,
        c="blue",
        s=15,
        alpha=0.5,
        label=f"Pareto Optimal Solutions (n={len(mean_returns)})",
    )

    # Highlight the 3 Key Points
    plt.scatter(
        min_risk_cvar,
        min_risk_ret,
        c="green",
        marker="*",
        s=200,
        label=f"1. Min Risk (CVaR={min_risk_cvar:.5f})",
        zorder=10,
    )

    plt.scatter(
        max_ret_cvar,
        max_ret_ret,
        c="red",
        marker="*",
        s=200,
        label=f"2. Max Return (Mean={max_ret_ret:.5f})",
        zorder=10,
    )

    plt.scatter(
        knee_cvar,
        knee_ret,
        c="orange",
        marker="X",
        s=150,
        label="3. Balanced/Knee Point",
        zorder=10,
    )

    # Labels and Titles (Matches Proposal Terminology)
    plt.title(
        "Hasil Optimisasi NSGA-II: Pareto Optimal Front (Mean vs. CVaR)", fontsize=16
    )
    plt.xlabel("Risiko (CVaR @ 95%) - Objektif f2 (Minimasi)", fontsize=12)
    plt.ylabel("Return (Mean Harian) - Objektif f1 (Maksimasi)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Axis formatting to show Percentages (e.g., 0.05 -> 5.00%)
    ax = plt.gca()
    formatter = FuncFormatter(lambda y, _: f"{y:.3%}")
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)

    # Saving the plot safely
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist
    output_path = os.path.join(output_dir, "pareto_front_plot.png")

    print("\nVisualisasi Pareto Front berhasil dibuat.")
    plt.savefig(output_path)
    print(f"Plot disimpan di '{output_path}'")
    plt.show()

    # ---------------------------------------------------------
    # Detailed Analysis Output
    # ---------------------------------------------------------
    print_portfolio_analysis(
        "1. Portofolio Risiko Minimum (Paling Konservatif)",
        tickers,
        min_risk_weights,
        min_risk_ret,
        min_risk_cvar,
    )
    print_portfolio_analysis(
        "3. Portofolio Titik Tengah (Balanced/Knee)",
        tickers,
        knee_weights,
        knee_ret,
        knee_cvar,
    )
    print_portfolio_analysis(
        "2. Portofolio Return Maksimum (Paling Agresif)",
        tickers,
        max_ret_weights,
        max_ret_ret,
        max_ret_cvar,
    )
