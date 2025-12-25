"""
Utility functions and configuration
"""
import os
import json
import zipfile
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration object
class Config:
    """Default configuration for the application"""
    
    # Differential Evolution bounds (alpha, beta, delta, gamma)
    DE_BOUNDS = [
        (0.001, 2.0),    # alpha
        (0.001, 30.0),   # beta
        (0.001, 30.0),   # delta
        (0.001, 2.0)     # gamma
    ]
    
    # Default weights for loss function
    DEFAULT_WEIGHTS = {
        'w_prey': 1.0,
        'w_pred': 3.0
    }
    
    # Default smoothing window
    DEFAULT_SMOOTHING_WINDOW = 3
    
    # Solver settings
    SOLVER_RTOL = 1e-6
    SOLVER_ATOL = 1e-8
    
    # DE settings
    DE_MAXITER = 250
    DE_POPSIZE = 25
    DE_TOL = 1e-6
    DE_SEED = 42
    
    # Visualization directory
    VIZ_DIR = "visualizations"
    
    # Plot numbering for report
    PLOT_NAMES = {
        '01_data_asli.png': 'Data Asli',
        '02_pola_osilasi.png': 'Pola Osilasi',
        '03_smoothing_tren.png': 'Smoothing Tren',
        '04_overlay_awal.png': 'Overlay Awal',
        '05_overlay_final.png': 'Overlay Final',
        '06_phase_portrait.png': 'Phase Portrait',
        '07_3d_trajectory.png': '3D Trajectory',
        '08_overlay_skala_asli.png': 'Overlay Skala Asli'
    }


def ensure_dir(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object to the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value} to float, using default {default}")
        return default


def zip_outputs(
    output_dir: str,
    output_filename: str = "report_pack.zip",
    include_patterns: Optional[list] = None
) -> str:
    """
    Create a zip file containing all outputs.
    
    Args:
        output_dir: Directory containing files to zip
        output_filename: Name of the zip file
        include_patterns: List of file patterns to include (e.g., ['*.png', '*.json'])
        
    Returns:
        Path to the created zip file
    """
    output_path = Path(output_dir)
    zip_path = output_path / output_filename
    
    if include_patterns is None:
        include_patterns = ['*.png', '*.json', '*.csv']
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for pattern in include_patterns:
            for file_path in output_path.glob(pattern):
                zipf.write(file_path, file_path.name)
                logger.info(f"Added {file_path.name} to zip")
    
    logger.info(f"Created zip file: {zip_path}")
    return str(zip_path)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file to dictionary."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

