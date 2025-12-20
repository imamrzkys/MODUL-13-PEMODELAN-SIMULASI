"""
Modul untuk semua fungsi visualisasi data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple

from src.utils import normalize_data, check_statsmodels

# Atur gaya plot agar sesuai dengan tema gelap Streamlit
plt.style.use('dark_background')

def plot_time_series(df: pd.DataFrame) -> Figure:
    """Plot deret waktu populasi Moose dan Serigala dalam satu sumbu."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Moose'], label='Moose', color='skyblue')
    ax.plot(df.index, df['Wolves'], label='Serigala', color='salmon')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Populasi')
    ax.set_title('Populasi Moose dan Serigala (1980-2019)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_time_series_twinx(df: pd.DataFrame) -> Figure:
    """Plot deret waktu dengan dua sumbu Y untuk Moose dan Serigala."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('Populasi Moose', color='skyblue')
    ax1.plot(df.index, df['Moose'], color='skyblue', label='Moose')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Populasi Serigala', color='salmon')
    ax2.plot(df.index, df['Wolves'], color='salmon', label='Serigala')
    ax2.tick_params(axis='y', labelcolor='salmon')

    ax1.set_title('Populasi Moose vs. Serigala (Sumbu Terpisah)')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_normalized(df: pd.DataFrame) -> Figure:
    """Plot data populasi yang sudah dinormalisasi (0-1)."""
    df_normalized = normalize_data(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_normalized.index, df_normalized['Moose'], label='Moose (Normal)', color='skyblue')
    ax.plot(df_normalized.index, df_normalized['Wolves'], label='Serigala (Normal)', color='salmon')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Populasi Ternormalisasi')
    ax.set_title('Populasi Ternormalisasi (0-1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_phase_diagram(df: pd.DataFrame) -> Figure:
    """Plot diagram fase yang menunjukkan hubungan Moose vs. Serigala."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(df['Moose'], df['Wolves'], marker='o', linestyle='-', color='cyan', alpha=0.7)
    ax.set_xlabel('Populasi Moose')
    ax.set_ylabel('Populasi Serigala')
    ax.set_title('Diagram Fase: Moose vs. Serigala')
    ax.grid(True, alpha=0.3)

    # Anotasi tahun setiap 4 langkah
    for i, year in enumerate(df.index):
        if i % 4 == 0:
            ax.text(df['Moose'].iloc[i], df['Wolves'].iloc[i], str(year), fontsize=9, alpha=0.8)

    fig.tight_layout()
    return fig

def plot_3d_trajectory(df: pd.DataFrame) -> Figure:
    """Plot trajektori 3D (Tahun, Moose, Serigala)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df.index, df['Moose'], df['Wolves'], color='magenta')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Populasi Moose')
    ax.set_zlabel('Populasi Serigala')
    ax.set_title('Trajektori 3D Populasi')
    ax.view_init(elev=20., azim=-35)
    fig.tight_layout()
    return fig

def plot_decomposition(df: pd.DataFrame, column: str, period: int) -> Tuple[Optional[Figure], str]:
    """Plot dekomposisi deret waktu (tren, siklus, residual)."""
    if not check_statsmodels():
        # Fallback manual jika statsmodels tidak ada
        rolling_mean = df[column].rolling(window=period, center=True).mean()
        detrended = df[column] - rolling_mean
        residual = detrended - detrended.rolling(window=2, center=True).mean() # Simple residual

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        df[column].plot(ax=ax1, title='Data Asli')
        rolling_mean.plot(ax=ax2, title='Tren (Moving Average)')
        detrended.plot(ax=ax3, title='Siklus (Data - Tren)')
        residual.plot(ax=ax4, title='Residual')
        fig.suptitle(f'Dekomposisi Manual untuk {column}', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, "`statsmodels` tidak ditemukan. Menggunakan dekomposisi manual dengan moving average."

    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df[column], model='additive', period=period)
    fig = result.plot()
    fig.set_size_inches(10, 12)
    fig.suptitle(f'Dekomposisi Pola untuk {column}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, f"Dekomposisi menggunakan `statsmodels` dengan periode {period}."

def plot_rolling_stats(df: pd.DataFrame, column: str, window: int) -> Figure:
    """Plot rata-rata dan standar deviasi bergulir."""
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df[column], label='Data Asli')
    ax.plot(rolling_mean.index, rolling_mean, label=f'Rata-rata Bergulir ({window} thn)', color='yellow')
    ax.plot(rolling_std.index, rolling_std, label=f'Std Dev Bergulir ({window} thn)', color='lime')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Populasi')
    ax.set_title(f'Statistik Bergulir untuk {column}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def run_adf_test(series: pd.Series) -> Tuple[float, float, str]:
    """Jalankan Uji Augmented Dickey-Fuller (ADF)."""
    if not check_statsmodels():
        return np.nan, np.nan, "`statsmodels` tidak ditemukan, tidak bisa menjalankan uji ADF."

    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    adf_stat, p_value = result[0], result[1]

    if p_value <= 0.05:
        interpretation = f"P-value ({p_value:.3f}) <= 0.05. Data kemungkinan stasioner."
    else:
        interpretation = f"P-value ({p_value:.3f}) > 0.05. Data kemungkinan tidak stasioner."

    return adf_stat, p_value, interpretation
