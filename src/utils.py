"""
Modul untuk fungsi-fungsi utilitas.
"""

import pandas as pd
from typing import Optional


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi data ke rentang 0-1 (Min-Max Scaling).

    Args:
        df (pd.DataFrame): DataFrame untuk dinormalisasi.

    Returns:
        pd.DataFrame: DataFrame yang sudah dinormalisasi.
    """
    return (df - df.min()) / (df.max() - df.min())


def check_statsmodels() -> bool:
    """Periksa apakah library statsmodels terinstal.

    Returns:
        bool: True jika terinstal, False jika tidak.
    """
    try:
        import statsmodels.api as sm
        return True
    except ImportError:
        return False
