"""
Gradient descent and stochastic gradient descent and its application
on linear regression using numpy.
"""
from argparse import ArgumentParser
import logging
from time import perf_counter
from typing import Any, Callable
import numpy as np
import matplotlib.pyplot as plt

from src.gradient_descent import minibatch_sgd, gradient_descent
from src.losses import l2_loss, l2_grad_theta
from src.helpers import make_regression_dataset, split_into_train_test
from src.metrics import mse, mae, r2_score

logging.basicConfig(level=logging.INFO)

def main():

    # argument parsing
    parser = ArgumentParser(prog="Linear regression using SGD",
                            description="A demonstration of linear regression using gradient descent in NumPy.",
                            epilog="Please report any bugs by raising an issue at https://github.com/debajyotid2/linear_regression_with_sgd.git")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=20, help="Number of features")
    parser.add_argument("--noise", type=float, default=2.0, help="Intensity of Gaussian noise to be added")
    parser.add_argument("--bias", type=float, default=-300.7, help="Bias term")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for gradient descent")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--seed", type=int, default=234, help="Random number seed")
    parser.add_argument("--max_iter", type=float, default=10000, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=0.001, help="Tolerance for convergence")

    args = parser.parse_args()
            
    # Random number seed
    np.random.seed(args.seed)
    
    # Making dataset and other parameters
    loss_fn = l2_loss
    grad_fn_theta = l2_grad_theta

    x, y = make_regression_dataset(args.n_samples, args.n_features, 
                                   args.bias, args.noise, args.seed)
    train_idx, test_idx = split_into_train_test(np.arange(x.shape[0]), 
                                                args.test_frac, args.seed)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Gradient descent
    theta = np.random.random(size=(args.n_features, 1))
    start = perf_counter()
    theta_gd, bias_gd, losses_gd = gradient_descent(x_train, y_train, theta, 
                                     loss_fn, grad_fn_theta, args.learning_rate, 
                                     args.max_iter, args.tol)
    logging.info(f"Gradient descent took {perf_counter()-start:.6f} seconds.")
    
    y_pred = x_test @ theta_gd + bias_gd
    logging.info(f"MSE = {mse(y_test, y_pred):.4f}")
    logging.info(f"MAE = {mae(y_test, y_pred):.4f}")
    logging.info(f"R2 = {r2_score(y_test, y_pred):.4f}")
    
    # Minibatch stochastic gradient descent
    theta = np.random.random(size=(args.n_features, 1))
    start = perf_counter()
    theta_mbsgd, bias_mbsgd, losses_mbsgd = minibatch_sgd(x_train, y_train, 
                                theta, args.batch_size, loss_fn, 
                                grad_fn_theta, args.learning_rate, 
                                args.max_iter, args.tol)
    logging.info(f"Minibatch stochastic gradient descent took {perf_counter()-start:.6f} seconds.")
    y_pred = x_test @ theta_mbsgd + bias_mbsgd
    logging.info(f"MSE = {mse(y_test, y_pred):.4f}")
    logging.info(f"MAE = {mae(y_test, y_pred):.4f}")
    logging.info(f"R2 = {r2_score(y_test, y_pred):.4f}")
    
    # Plot losses
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 14
    plt.plot(np.arange(len(losses_gd))+1, 
             losses_gd, label="GD")
    plt.plot(np.arange(len(losses_mbsgd))+1, 
             losses_mbsgd, label="MBSGD")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.show()

if __name__=="__main__":
    main()
