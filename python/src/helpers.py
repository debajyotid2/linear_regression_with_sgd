"""
Helper functions.
"""
from typing import Any
import numpy as np

def make_regression_dataset(n_samples: int,
                            n_features: int,
                            bias: float = 0.0,
                            noise_intensity: float = 0.0,
                            seed: int = 42) -> \
                            tuple[np.ndarray[Any, Any],
                                  np.ndarray[Any, Any]]:
    """
    Create a dataset for solving linear regression problems.
    """
    rng = np.random.default_rng(seed)
    means = 10 * rng.random(size=(n_features,))
    stddevs = 10 * rng.random(size=(n_features,))
    theta_true = 50 * rng.random(size=(n_features, 1)) - 25
    x = rng.normal(loc=means, scale=stddevs, 
                         size=(n_samples, n_features))
    noise_arr = noise_intensity * rng.normal(size=(n_samples, 1))
    y = x @ theta_true + bias + noise_arr
    
    return x, y

def split_into_train_test(idx: np.ndarray[Any, Any],
                          test_frac: float,
                          seed: int = 42) -> \
                          tuple[np.ndarray[Any, Any],
                                np.ndarray[Any, Any]]:
    """
    Creates indices by drawing a random sample from
    an array of indices.
    """
    rng = np.random.default_rng(seed)
    n_test = int(test_frac * idx.shape[0])
    test_idx = rng.choice(idx, size=(n_test,))
    train_idx = np.delete(idx, test_idx)

    return train_idx, test_idx
