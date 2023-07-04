"""
Gradient descent algorithms.
"""
import logging
from typing import Any, Callable
import numpy as np

logging.getLogger(__name__)

LOSS_INTERVAL = 100

def forward(x: np.ndarray[Any, Any],
            theta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Evaluates the linear function f(x).
    """
    return x @ theta

def update_theta(x: np.ndarray[Any, Any],
             y: np.ndarray[Any, Any],
             theta: np.ndarray[Any, Any],
             grad_fn: Callable[Any, Any],
             eta: float) -> np.ndarray[Any, Any]: 
    """
    Determines the new value of theta using the gradient 
    descent method.
    """
    return theta - (eta / y.shape[0]) * grad_fn(x, y, theta)

def gradient_descent(x: np.ndarray[Any, Any],
                     y: np.ndarray[Any, Any],
                     theta_init: np.ndarray[Any, Any],
                     loss_fn: Callable[Any, Any],
                     grad_fn_theta: Callable[Any, Any],
                     eta: float,
                     max_iter: int = 100000,
                     tol: int = 1e-5) ->\
                tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any]]:
    """
    Gradient descent algorithm.
    """
    converged = False
    losses = []
    theta = theta_init
    
    y_vals = y.copy()
    y_offset = np.mean(y, axis=0)
    y_vals -= y_offset
    
    x_vals = x.copy()
    x_offset = np.mean(x, axis=0)
    x_vals -= x_offset

    for i in range(max_iter):
        y_pred = forward(x_vals, theta)
        loss = loss_fn(y_vals, y_pred)
        if loss < tol:
            converged = True
            break
        theta = update_theta(x_vals, y_vals, theta, grad_fn_theta, eta)
        if (i+1)%LOSS_INTERVAL == 0:
            logging.info(f"{i+1}: loss = {loss:.4f}")
            losses.append(loss)
    if converged:
        logging.info(f"Converged in {i+1} iterations.")
    else:
        logging.info("Gradient descent did not converge.")
    
    bias = y_offset - x_offset @ theta
    return theta, bias, np.asarray(losses, dtype=np.float32)

def minibatch_sgd(x: np.ndarray[Any, Any],
                  y: np.ndarray[Any, Any],
                  theta_init: np.ndarray[Any, Any],
                  batch_size: int,
                  loss_fn: Callable[Any, Any],
                  grad_fn_theta: Callable[Any, Any],
                  eta: float,
                  max_iter: int = 100000,
                  tol: int = 1e-5) ->\
                tuple[np.ndarray[Any, Any], float, np.ndarray[Any, Any]]:
    """
    Minibatch stochastic gradient descent algorithm.
    """
    converged = False
    losses = []
    theta = theta_init
    
    y_vals = y.copy()
    y_offset = np.mean(y, axis=0)
    y_vals -= y_offset
    
    x_vals = x.copy()
    x_offset = np.mean(x, axis=0)
    x_vals -= x_offset

    for i in range(max_iter):
        idx = np.random.randint(low=0, high=x_vals.shape[0], size=(batch_size,))
        x_batch, y_batch = x_vals[idx], y_vals[idx]
        y_pred = forward(x_batch, theta)
        loss = loss_fn(y_batch, y_pred)
        if loss < tol:
            converged = True
            break
        theta = update_theta(x_batch, y_batch, theta, grad_fn_theta, eta)
        if (i+1)%LOSS_INTERVAL == 0:
            logging.info(f"{i+1}: loss = {loss:.4f}")
            losses.append(loss)
    if converged:
        logging.info(f"Converged in {i+1} iterations.")
    else:
        logging.info("Stochastic gradient descent did not converge.")
    
    bias = y_offset - x_offset @ theta
    return theta, bias, np.asarray(losses, dtype=np.float32)
