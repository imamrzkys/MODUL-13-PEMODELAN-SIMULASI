"""
Modul untuk simulasi dan fitting model Lotka-Volterra.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

def lotka_volterra(t, y, alpha, beta, delta, gamma):
    """Mendefinisikan sistem persamaan diferensial Lotka-Volterra."""
    # y[0] = Moose (mangsa), y[1] = Serigala (predator)
    dxdt = alpha * y[0] - beta * y[0] * y[1]
    dydt = delta * y[0] * y[1] - gamma * y[1]
    return [dxdt, dydt]


def run_simulation(params: Dict[str, float], initial_conditions: Tuple[float, float], t_eval: np.ndarray) -> pd.DataFrame:
    """Menjalankan simulasi Lotka-Volterra dengan parameter yang diberikan."""
    alpha, beta, delta, gamma = params['alpha'], params['beta'], params['delta'], params['gamma']

    solution = solve_ivp(
        fun=lotka_volterra,
        t_span=[t_eval[0], t_eval[-1]],
        y0=initial_conditions,
        t_eval=t_eval,
        args=(alpha, beta, delta, gamma)
    )

    sim_df = pd.DataFrame({
        'Moose_sim': solution.y[0],
        'Wolves_sim': solution.y[1]
    }, index=t_eval.astype(int))
    return sim_df


def fit_parameters(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Mencari parameter Lotka-Volterra terbaik menggunakan least squares."""
    initial_params = [0.5, 0.02, 0.01, 0.8]  # Tebakan awal
    t_eval = np.arange(len(df))
    y_true = df[['Moose', 'Wolves']].values

    def residuals(params):
        alpha, beta, delta, gamma = params
        sim_result = solve_ivp(
            fun=lotka_volterra,
            t_span=[t_eval[0], t_eval[-1]],
            y0=y_true[0],
            t_eval=t_eval,
            args=(alpha, beta, delta, gamma)
        ).y.T
        # Hitung residual (perbedaan antara data asli dan simulasi)
        return (y_true - sim_result).flatten()

    result = least_squares(residuals, initial_params, bounds=(0, np.inf))
    fitted_params = {
        'alpha': result.x[0],
        'beta': result.x[1],
        'delta': result.x[2],
        'gamma': result.x[3]
    }

    # Hitung error (RMSE)
    final_residuals = residuals(result.x).reshape(y_true.shape)
    rmse_moose = np.sqrt(np.mean(final_residuals[:, 0]**2))
    rmse_wolves = np.sqrt(np.mean(final_residuals[:, 1]**2))
    errors = {'RMSE_Moose': rmse_moose, 'RMSE_Wolves': rmse_wolves}

    return fitted_params, errors


def plot_simulation_vs_data(df: pd.DataFrame, sim_df: pd.DataFrame) -> plt.Figure:
    """Plot perbandingan data asli dengan hasil simulasi."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Moose'], 'o-', label='Data Moose', color='skyblue')
    ax.plot(df.index, df['Wolves'], 'o-', label='Data Serigala', color='salmon')
    ax.plot(df.index, sim_df['Moose_sim'], '--', label='Simulasi Moose', color='cyan')
    ax.plot(df.index, sim_df['Wolves_sim'], '--', label='Simulasi Serigala', color='red')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Populasi')
    ax.set_title('Data Asli vs. Hasil Simulasi Lotka-Volterra')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_fit_vs_data(df: pd.DataFrame, fitted_params: Dict[str, float]) -> plt.Figure:
    """Plot perbandingan data asli dengan model yang sudah di-fit."""
    t_eval = np.arange(len(df))
    initial_conditions = (df['Moose'].iloc[0], df['Wolves'].iloc[0])
    
    sim_df = run_simulation(fitted_params, initial_conditions, t_eval)
    sim_df.index = df.index # Sesuaikan index tahun

    return plot_simulation_vs_data(df, sim_df)
