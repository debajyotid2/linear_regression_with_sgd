"""
Loss functions.
"""
from typing import Any, Callable
import numpy as np

def l2_grad_theta(x: np.ndarray[Any, Any],
                  y: np.ndarray[Any, Any],
                  theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Gradient of the mean squared error loss function.
    """
    return 2 * x.T @ (x @ theta - y)

def l2_grad_with_reg(x: np.ndarray[Any, Any],
                    y: np.ndarray[Any, Any],
                    theta: np.ndarray[Any, Any],
                    reg_grad_fn: Callable[Any, Any],
                    lamda: float) -> np.ndarray[Any, Any]:
    """
    Gradient of the mean squared error loss function with
    regularization.
    """
    return 2 * (x.T @ (x @ theta - y)) + lamda * reg_grad_fn(theta)

def l2_loss(y: np.ndarray[Any, Any],
            y_pred: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Mean squared error loss function, aka L2 loss.
    """
    diff = y - y_pred
    return (np.dot(diff.T, diff))[0, 0] / y.shape[0]

def l2_reg(theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    L2 regularization.
    """
    return (theta.T @ theta)[0, 0]

def l2_reg_grad_theta(theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Gradient of the L2 regularization term with respect to theta.
    """
    return 2 * theta
