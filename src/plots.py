"""
Plotting functions for visualizations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, Optional, Dict
import logging

# Try to import plotly, but make it optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .utils import ensure_dir, Config

logger = logging.getLogger(__name__)

if not PLOTLY_AVAILABLE:
    logger.warning("Plotly not available, 3D plots will use matplotlib")


def save_fig(fig: Figure, filename: str, directory: str = None) -> str:
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        directory: Output directory (default: Config.VIZ_DIR)
        
    Returns:
        Full path to saved file
    """
    if directory is None:
        directory = Config.VIZ_DIR
    
    ensure_dir(directory)
    filepath = f"{directory}/{filename}"
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    logger.info(f"Saved figure: {filepath}")
    return filepath


def plot_raw_data(
    df: pd.DataFrame,
    title: str = "Data Asli: Moose vs Wolves",
    save: bool = True,
    filename: str = "01_data_asli.png"
) -> Figure:
    """
    Plot raw time series data.
    
    Args:
        df: DataFrame with 'prey' and 'predator' columns, year as index
        title: Plot title
        save: Whether to save the figure
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['prey'], 'o-', label='Moose (Prey)', linewidth=2, markersize=6)
    ax.plot(df.index, df['predator'], 's-', label='Wolves (Predator)', linewidth=2, markersize=6)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_oscillation_pattern(
    df: pd.DataFrame,
    title: str = "Pola Osilasi (Min-Max Scaling)",
    save: bool = True,
    filename: str = "02_pola_osilasi.png"
) -> Figure:
    """
    Plot min-max scaled data showing oscillation pattern.
    
    Args:
        df: DataFrame with scaled columns
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute min-max scaling
    prey_min, prey_max = df['prey'].min(), df['prey'].max()
    pred_min, pred_max = df['predator'].min(), df['predator'].max()
    
    prey_mm = (df['prey'] - prey_min) / (prey_max - prey_min)
    pred_mm = (df['predator'] - pred_min) / (pred_max - pred_min)
    
    ax.plot(df.index, prey_mm, 'o-', label='Prey (MinMax)', linewidth=2, markersize=5)
    ax.plot(df.index, pred_mm, 's-', label='Predator (MinMax)', linewidth=2, markersize=5)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Scaled Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_smoothing_trend(
    df: pd.DataFrame,
    window: int = 3,
    title: str = "Smoothing untuk melihat tren osilasi",
    save: bool = True,
    filename: str = "03_smoothing_tren.png"
) -> Figure:
    """
    Plot raw data with moving average smoothing.
    
    Args:
        df: DataFrame with 'prey' and 'predator' columns
        window: Moving average window size
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prey_ma = pd.Series(df['prey'].values).rolling(window, center=True).mean()
    pred_ma = pd.Series(df['predator'].values).rolling(window, center=True).mean()
    
    ax.plot(df.index, df['prey'], alpha=0.35, label='Prey raw', linewidth=1.5)
    ax.plot(df.index, prey_ma, label=f'Prey MA({window})', linewidth=2.5)
    
    ax.plot(df.index, df['predator'], alpha=0.35, label='Pred raw', linewidth=1.5)
    ax.plot(df.index, pred_ma, label=f'Pred MA({window})', linewidth=2.5)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_overlay_initial(
    df_scaled: pd.DataFrame,
    sim_prey: np.ndarray,
    sim_pred: np.ndarray,
    title: str = "Overlay Awal (Sebelum Tuning)",
    save: bool = True,
    filename: str = "04_overlay_awal.png"
) -> Figure:
    """
    Plot overlay of data vs initial simulation.
    
    Args:
        df_scaled: DataFrame with scaled data columns
        sim_prey: Simulated prey values
        sim_pred: Simulated predator values
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_scaled.index, df_scaled['prey_scaled'], '--', 
            label='Prey data (scaled)', alpha=0.75, linewidth=2)
    ax.plot(df_scaled.index, df_scaled['predator_scaled'], '--', 
            label='Pred data (scaled)', alpha=0.75, linewidth=2)
    
    ax.plot(df_scaled.index, sim_prey, label='Prey sim (init)', linewidth=2.5)
    ax.plot(df_scaled.index, sim_pred, label='Pred sim (init)', linewidth=2.5)
    
    ax.set_xlabel('Time index (per tahun)', fontsize=12)
    ax.set_ylabel('Scaled Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_overlay_final(
    df_scaled: pd.DataFrame,
    sim_prey: np.ndarray,
    sim_pred: np.ndarray,
    title: str = "Overlay FINAL: Data Asli vs Simulasi Lotka–Volterra (Setelah Tuning)",
    save: bool = True,
    filename: str = "05_overlay_final.png"
) -> Figure:
    """
    Plot overlay of data vs best-fit simulation.
    
    Args:
        df_scaled: DataFrame with scaled data columns
        sim_prey: Simulated prey values
        sim_pred: Simulated predator values
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_scaled.index, df_scaled['prey_scaled'], '--', 
            label='Prey data (scaled)', alpha=0.75, linewidth=2)
    ax.plot(df_scaled.index, df_scaled['predator_scaled'], '--', 
            label='Pred data (scaled)', alpha=0.75, linewidth=2)
    
    ax.plot(df_scaled.index, sim_prey, label='Prey sim (best)', linewidth=2.5)
    ax.plot(df_scaled.index, sim_pred, label='Pred sim (best)', linewidth=2.5)
    
    ax.set_xlabel('Time index (per tahun)', fontsize=12)
    ax.set_ylabel('Scaled Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_phase_portrait(
    data_prey: np.ndarray,
    data_pred: np.ndarray,
    sim_prey: np.ndarray,
    sim_pred: np.ndarray,
    title: str = "Phase Portrait: Data vs Model",
    save: bool = True,
    filename: str = "06_phase_portrait.png"
) -> Figure:
    """
    Plot phase portrait (prey vs predator).
    
    Args:
        data_prey: Observed prey values
        data_pred: Observed predator values
        sim_prey: Simulated prey values
        sim_pred: Simulated predator values
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(data_prey, data_pred, 'o-', alpha=0.6, label='Data (scaled)', 
            linewidth=2, markersize=6)
    ax.plot(sim_prey, sim_pred, '-', linewidth=2.5, label='Simulasi (best)')
    
    ax.set_xlabel('Prey', fontsize=12)
    ax.set_ylabel('Predator', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_3d_trajectory_matplotlib(
    t: np.ndarray,
    prey: np.ndarray,
    predator: np.ndarray,
    title: str = "3D Trajectory: time–prey–predator (Simulasi)",
    save: bool = True,
    filename: str = "07_3d_trajectory.png"
) -> Figure:
    """
    Plot 3D trajectory using matplotlib (fallback).
    
    Args:
        t: Time values
        prey: Prey values
        predator: Predator values
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(t, prey, predator, linewidth=2.5)
    ax.set_xlabel('Time index', fontsize=11)
    ax.set_ylabel('Prey (sim)', fontsize=11)
    ax.set_zlabel('Predator (sim)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_3d_trajectory_plotly(
    t: np.ndarray,
    prey: np.ndarray,
    predator: np.ndarray,
    title: str = "3D Trajectory: time–prey–predator (Simulasi)"
) -> Optional[go.Figure]:
    """
    Plot 3D trajectory using Plotly (for Streamlit).
    
    Args:
        t: Time values
        prey: Prey values
        predator: Predator values
        title: Plot title
        
    Returns:
        Plotly figure or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure(data=go.Scatter3d(
        x=t,
        y=prey,
        z=predator,
        mode='lines',
        line=dict(color='blue', width=4),
        name='Trajectory'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time index',
            yaxis_title='Prey (sim)',
            zaxis_title='Predator (sim)'
        ),
        width=800,
        height=600
    )
    
    return fig


def plot_overlay_real_scale(
    df: pd.DataFrame,
    sim_prey_real: np.ndarray,
    sim_pred_real: np.ndarray,
    title: str = "Overlay pada Skala Asli (Real Population)",
    save: bool = True,
    filename: str = "08_overlay_skala_asli.png"
) -> Figure:
    """
    Plot overlay on real population scale.
    
    Args:
        df: DataFrame with original 'prey' and 'predator' columns
        sim_prey_real: Simulated prey values on real scale
        sim_pred_real: Simulated predator values on real scale
        title: Plot title
        save: Whether to save
        filename: Output filename
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df.index, df['prey'], '--', label='Prey data (real)', 
            alpha=0.8, linewidth=2)
    ax.plot(df.index, df['predator'], '--', label='Pred data (real)', 
            alpha=0.8, linewidth=2)
    
    ax.plot(df.index, sim_prey_real, label='Prey sim (real-scale)', linewidth=2.5)
    ax.plot(df.index, sim_pred_real, label='Pred sim (real-scale)', linewidth=2.5)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    if save:
        save_fig(fig, filename)
    
    return fig


def plot_residuals(
    data_prey: np.ndarray,
    data_pred: np.ndarray,
    sim_prey: np.ndarray,
    sim_pred: np.ndarray,
    years: np.ndarray,
    title: str = "Residuals: Data - Simulation"
) -> Figure:
    """
    Plot residuals for prey and predator.
    
    Args:
        data_prey: Observed prey values
        data_pred: Observed predator values
        sim_prey: Simulated prey values
        sim_pred: Simulated predator values
        years: Year values for x-axis
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    res_prey = data_prey - sim_prey
    res_pred = data_pred - sim_pred
    
    ax1.plot(years, res_prey, 'o-', color='blue', linewidth=2, markersize=5)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Residual (Prey)', fontsize=12)
    ax1.set_title(f'{title} - Prey', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(years, res_pred, 's-', color='orange', linewidth=2, markersize=5)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Residual (Predator)', fontsize=12)
    ax2.set_title(f'{title} - Predator', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

