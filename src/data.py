"""
Data loading and processing functions
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_data(filepath: str = "wolf_moose_nps.csv") -> pd.DataFrame:
    """
    Load and prepare the predator-prey dataset.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with columns: year, prey, predator
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded data from {filepath}: {len(df)} rows")
        return clean_columns(df)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names and standardize format.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame with standardized columns
    """
    # Strip and lowercase column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Check required columns
    required = {"year", "wolves", "moose"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ada: {missing}. "
            f"Kolom yang ada: {df.columns.tolist()}"
        )
    
    # Select and rename columns
    df_clean = df[["year", "moose", "wolves"]].copy()
    df_clean = df_clean.rename(columns={"moose": "prey", "wolves": "predator"})
    
    # Sort by year and set as index
    df_clean = df_clean.sort_values("year").reset_index(drop=True)
    df_clean = df_clean.set_index("year")
    
    # Ensure numeric types
    df_clean['prey'] = pd.to_numeric(df_clean['prey'], errors='coerce')
    df_clean['predator'] = pd.to_numeric(df_clean['predator'], errors='coerce')
    
    # Remove any rows with NaN
    df_clean = df_clean.dropna()
    
    logger.info(f"Cleaned data: {len(df_clean)} rows, years {df_clean.index.min()}-{df_clean.index.max()}")
    return df_clean


def add_scaling(df: pd.DataFrame, method: str = "max") -> Tuple[pd.DataFrame, float]:
    """
    Add scaled columns to dataframe.
    
    Args:
        df: DataFrame with 'prey' and 'predator' columns
        method: Scaling method ('max' or 'minmax')
        
    Returns:
        Tuple of (DataFrame with scaled columns, scale_factor)
    """
    df_scaled = df.copy()
    
    if method == "max":
        # Scale by maximum value (as in notebook)
        scale_factor = max(df['prey'].max(), df['predator'].max())
        df_scaled['prey_scaled'] = df_scaled['prey'] / scale_factor
        df_scaled['predator_scaled'] = df_scaled['predator'] / scale_factor
    elif method == "minmax":
        # Min-max scaling
        prey_min, prey_max = df['prey'].min(), df['prey'].max()
        pred_min, pred_max = df['predator'].min(), df['predator'].max()
        
        df_scaled['prey_scaled'] = (df_scaled['prey'] - prey_min) / (prey_max - prey_min)
        df_scaled['predator_scaled'] = (df_scaled['predator'] - pred_min) / (pred_max - pred_min)
        scale_factor = 1.0  # Not used for minmax
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    logger.info(f"Added scaling (method={method}, scale_factor={scale_factor})")
    return df_scaled, scale_factor


def filter_year_range(
    df: pd.DataFrame,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None
) -> pd.DataFrame:
    """
    Filter dataframe by year range.
    
    Args:
        df: DataFrame with year as index
        year_min: Minimum year (inclusive)
        year_max: Maximum year (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    if year_min is not None:
        df_filtered = df_filtered[df_filtered.index >= year_min]
    if year_max is not None:
        df_filtered = df_filtered[df_filtered.index <= year_max]
    
    logger.info(f"Filtered to {len(df_filtered)} rows (years {df_filtered.index.min()}-{df_filtered.index.max()})")
    return df_filtered


def get_smoothed_series(
    df: pd.DataFrame,
    window: int = 3,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Apply moving average smoothing to time series.
    
    Args:
        df: DataFrame with time series data
        window: Window size for moving average
        columns: Columns to smooth (default: ['prey', 'predator'])
        
    Returns:
        DataFrame with smoothed columns
    """
    if columns is None:
        columns = ['prey', 'predator']
    
    df_smooth = df.copy()
    for col in columns:
        if col in df.columns:
            df_smooth[f'{col}_ma'] = pd.Series(df[col].values).rolling(
                window, center=True
            ).mean()
    
    return df_smooth

