"""
Regression metrics.
"""
from typing import Any
import numpy as np

def mse(y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any]) -> float:
    """
    Mean squared error.
    """
    diff = y_true - y_pred
    return (diff.T @ diff)[0, 0] / y_true.shape[0]

def mae(y_true: np.ndarray[Any, Any],
        y_pred: np.ndarray[Any, Any]) -> float:
    """
    Mean absolute error.
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray[Any, Any],
             y_pred: np.ndarray[Any, Any]) -> float:
    """
    Coefficient of determination.
    $$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$
    """
    y_mean = np.mean(y_true)
    diff = y_true - y_pred
    ss_res = (diff.T @ diff)[0, 0]
    ss_tot = ((y_true - y_mean).T @ (y_true - y_mean))[0, 0]
    return 1.0 - ss_res/ss_tot
