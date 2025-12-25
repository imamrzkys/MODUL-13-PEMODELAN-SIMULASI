"""
Lotka-Volterra model functions
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional, Callable
import logging

from .utils import Config

logger = logging.getLogger(__name__)


def lv_rhs(t: float, z: np.ndarray, alpha: float, beta: float, delta: float, gamma: float) -> np.ndarray:
    """
    Lotka-Volterra right-hand side function for ODE system.
    
    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y
    
    Args:
        t: Time (not used, but required by solve_ivp)
        z: State vector [x, y] where x=prey, y=predator
        alpha: Prey growth rate
        beta: Predation rate
        delta: Predator growth efficiency
        gamma: Predator death rate
        
    Returns:
        Array [dx/dt, dy/dt]
    """
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return np.array([dxdt, dydt])


def simulate_lv(
    params: Dict[str, float],
    initial_conditions: Tuple[float, float],
    t_eval: np.ndarray,
    method: str = 'RK45',
    rtol: float = None,
    atol: float = None
) -> Optional[np.ndarray]:
    """
    Simulate Lotka-Volterra model using solve_ivp.
    
    Args:
        params: Dictionary with keys 'alpha', 'beta', 'delta', 'gamma'
        initial_conditions: Tuple (x0, y0) initial prey and predator populations
        t_eval: Time points to evaluate solution
        method: ODE solver method (default: 'RK45')
        rtol: Relative tolerance (default: from Config)
        atol: Absolute tolerance (default: from Config)
        
    Returns:
        Array of shape (len(t_eval), 2) with [prey, predator] columns, or None if failed
    """
    if rtol is None:
        rtol = Config.SOLVER_RTOL
    if atol is None:
        atol = Config.SOLVER_ATOL
    
    try:
        sol = solve_ivp(
            lv_rhs,
            (t_eval[0], t_eval[-1]),
            initial_conditions,
            args=(params['alpha'], params['beta'], params['delta'], params['gamma']),
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol
        )
        
        if not sol.success:
            logger.warning(f"Solver failed: {sol.message}")
            return None
        
        if sol.y.shape[1] != len(t_eval):
            logger.warning(f"Solution shape mismatch: expected {len(t_eval)}, got {sol.y.shape[1]}")
            return None
        
        # Check for invalid values
        if np.any(~np.isfinite(sol.y)):
            logger.warning("Solution contains non-finite values")
            return None
        
        # Return as (N, 2) array
        return sol.y.T
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaled: bool = True
) -> Dict[str, float]:
    """
    Compute MSE and MAE metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        scaled: Whether data is scaled (for reporting)
        
    Returns:
        Dictionary with 'mse' and 'mae' keys
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'scaled': scaled
    }


def fit_de_params(
    prey_data: np.ndarray,
    pred_data: np.ndarray,
    t_eval: np.ndarray,
    initial_conditions: Tuple[float, float],
    bounds: Optional[list] = None,
    weights: Optional[Dict[str, float]] = None,
    maxiter: int = None,
    popsize: int = None,
    seed: int = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Fit Lotka-Volterra parameters using Differential Evolution.
    
    Args:
        prey_data: Observed prey values
        pred_data: Observed predator values
        t_eval: Time points
        initial_conditions: Initial (x0, y0)
        bounds: Parameter bounds [(min, max), ...] for [alpha, beta, delta, gamma]
        weights: Dictionary with 'w_prey' and 'w_pred' keys
        maxiter: Maximum iterations for DE
        popsize: Population size for DE
        seed: Random seed
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary with best parameters {'alpha', 'beta', 'delta', 'gamma'}
    """
    if bounds is None:
        bounds = Config.DE_BOUNDS
    
    if weights is None:
        weights = Config.DEFAULT_WEIGHTS
    
    if maxiter is None:
        maxiter = Config.DE_MAXITER
    
    if popsize is None:
        popsize = Config.DE_POPSIZE
    
    if seed is None:
        seed = Config.DE_SEED
    
    w_prey = weights.get('w_prey', 1.0)
    w_pred = weights.get('w_pred', 3.0)
    
    def loss_function(params):
        """Loss function for optimization"""
        p_dict = {
            'alpha': params[0],
            'beta': params[1],
            'delta': params[2],
            'gamma': params[3]
        }
        
        sim_results = simulate_lv(p_dict, initial_conditions, t_eval)
        if sim_results is None:
            return 1e9
        
        xs, ys = sim_results[:, 0], sim_results[:, 1]
        
        # Check for invalid values
        if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
            return 1e9
        
        mse_prey = mean_squared_error(prey_data, xs)
        mse_pred = mean_squared_error(pred_data, ys)
        
        loss = w_prey * mse_prey + w_pred * mse_pred
        
        # Call progress callback if provided
        if progress_callback:
            try:
                progress_callback(loss)
            except:
                pass
        
        return loss
    
    logger.info(f"Starting DE optimization (maxiter={maxiter}, popsize={popsize})")
    
    result = differential_evolution(
        loss_function,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        tol=Config.DE_TOL,
        seed=seed,
        polish=True
    )
    
    best_params = {
        'alpha': result.x[0],
        'beta': result.x[1],
        'delta': result.x[2],
        'gamma': result.x[3]
    }
    
    logger.info(f"DE completed. Best loss: {result.fun:.6f}")
    logger.info(f"Best params: {best_params}")
    
    return best_params


def compute_equilibrium(alpha: float, beta: float, delta: float, gamma: float) -> Tuple[float, float]:
    """
    Compute equilibrium point for Lotka-Volterra system.
    
    Equilibrium: (gamma/delta, alpha/beta)
    
    Args:
        alpha, beta, delta, gamma: Model parameters
        
    Returns:
        Tuple (x_eq, y_eq) equilibrium point
    """
    if delta == 0 or beta == 0:
        return (0.0, 0.0)
    
    x_eq = gamma / delta
    y_eq = alpha / beta
    
    return (x_eq, y_eq)

