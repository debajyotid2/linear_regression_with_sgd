// Stochastic gradient descent

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix.h"
#include "losses.h"
#include "stats.h"
#include "sgd.h"

// Iteration interval at which loss is recorded
const unsigned int LOSS_INTERVAL = 100;

// Initialize SGDResult object
void init_sgdresult(SGDResult* result,
                    unsigned int n_iter,
                    unsigned int n_features,
                    unsigned int seed)
{
    result->converged = false;
    result->n_iter = n_iter;
    result->bias = 0.0;
    result->losses = (double *)calloc(n_iter, sizeof(double));
    result->theta_sol = mat_create(n_features, 1);
    mat_fill_random(&(result->theta_sol), seed);
}

// Destroy SGDResult object
void destroy_sgdresult(SGDResult* result)
{
    if (result->losses!=NULL)
        free(result->losses);
    mat_destroy(&(result->theta_sol));
    result->losses = NULL;
}

// Forward method, y = x theta
void forward(Matrix* x, Matrix* theta, Matrix* y_pred)
{
    mat_mul_inplace(x, false, theta, false, y_pred);
}

// Backward method, theta := theta - 2 * eta / N * grad(x, y, theta)
void backward(Matrix* x, Matrix* y, 
              Matrix* theta, double eta,
              grad_fn_type grad_fn)
{
    Matrix grad = grad_fn(x, y, theta);
    mat_scale(&grad, 2.0 * eta /(double)(y->nrows));
    mat_sub(theta, &grad);

    mat_destroy(&grad);
}

// Gradient descent
SGDResult gradient_descent(
            Matrix* x, Matrix* y,
            double learning_rate,
            loss_fn_type loss_fn,
            grad_fn_type grad_fn,
            unsigned int n_iter,
            double tol,
            unsigned int seed)
{
    SGDResult result;
    unsigned int i = 0;
    double loss = 0.0;

    // Initialize result object
    init_sgdresult(&result, n_iter, x->ncols, seed);

    Matrix x_copy = mat_copy(x);
    Matrix y_copy = mat_copy(y);
    Matrix x_offset_theta_prod;
    Matrix bias_full;
    
    // y_pred for storing predictions
    Matrix y_pred = mat_create(y->nrows, y->ncols);
    mat_fill(&y_pred, 0.0);
    
    // Find the mean of x and y to center them and remove bias
    Matrix x_offset = stats_mean(x, 0);
    Matrix y_offset = stats_mean(y, 0);

    // Center x and y matrix to reduce problem from 
    // y = x theta + b to y = x theta
    mat_vec_sub(&x_copy, &x_offset);
    mat_vec_sub(&y_copy, &y_offset);

    // Gradient descent algorithm
    for (i=0; i<n_iter; i++)
    {
        // Update loss
        forward(&x_copy, &(result.theta_sol), &y_pred);
        loss = loss_fn(&y_copy, &y_pred);
        
        // Check convergence
        if (loss < tol)
        {
            result.converged = true;
            break;
        }
    
        // Print loss
        if ((i+1)%LOSS_INTERVAL == 0)
        {
            printf("It. %u, loss = %.4f\n", i+1, loss);
            result.losses[i] = loss;
        }
        
        // Update theta
        backward(&x_copy, &y_copy, 
                 &(result.theta_sol), learning_rate, grad_fn);
    }

    if (result.converged)
        printf("Converged in %u iterations.\n", i+1);

    // Calculate predicted bias
    // bias = y_offset - x_offset theta
    x_offset_theta_prod = mat_mul(&x_offset, false, 
                                &(result.theta_sol), false);
    mat_sub(&y_offset, &x_offset_theta_prod);
    bias_full = stats_mean(&y_offset, 2);
    result.bias = bias_full.data[0];
    
    // Destroy local matrices
    mat_destroy(&x_copy);
    mat_destroy(&y_copy);
    mat_destroy(&x_offset);
    mat_destroy(&y_offset);
    mat_destroy(&x_offset_theta_prod);
    mat_destroy(&y_pred);
    mat_destroy(&bias_full);

    // Truncate loss array
    if (i>LOSS_INTERVAL)
        result.losses = (double *)realloc(result.losses, 
                        ((i+1)/LOSS_INTERVAL) * sizeof(double));

    return result;
}

// Stochastic gradient descent
SGDResult stochastic_gradient_descent(
            Matrix* x, Matrix* y,
            unsigned int batch_size, 
            double learning_rate,
            loss_fn_type loss_fn,
            grad_fn_type grad_fn,
            unsigned int n_iter,
            double tol,
            unsigned int seed)
{
    SGDResult result;
    unsigned int i = 0;
    double loss = 0.0;
    IntMatrix idxs = intmat_create(batch_size, 1);

    // Initialize result object
    init_sgdresult(&result, n_iter, x->ncols, seed);

    Matrix x_copy = mat_copy(x);
    Matrix y_copy = mat_copy(y);
    Matrix x_batch = mat_create(batch_size, x->ncols);
    Matrix y_batch = mat_create(batch_size, y->ncols);
    Matrix x_offset_theta_prod;
    Matrix bias_full;
    
    // y_pred for storing predictions
    Matrix y_pred = mat_create(batch_size, y->ncols);
    mat_fill(&y_pred, 0.0);
    
    // Find the mean of x and y to center them and remove bias
    Matrix x_offset = stats_mean(x, 0);
    Matrix y_offset = stats_mean(y, 0);

    // Center x and y matrix to reduce problem from 
    // y = x theta + b to y = x theta
    mat_vec_sub(&x_copy, &x_offset);
    mat_vec_sub(&y_copy, &y_offset);

    // Minibatch Stochastic Gradient descent algorithm
    for (i=0; i<n_iter; i++)
    {
        // Generate batch idxs
        intmat_fill_random(&idxs, 0, y->nrows, false, seed);

        // Gather batch elements
        mat_gather(&x_copy, &x_batch, &idxs, 0);
        mat_gather(&y_copy, &y_batch, &idxs, 0);

        // Update loss
        forward(&x_batch, &(result.theta_sol), &y_pred);
        loss = loss_fn(&y_batch, &y_pred);
        
        // Check convergence
        if (loss < tol)
        {
            result.converged = true;
            break;
        }
    
        // Print loss
        if ((i+1)%LOSS_INTERVAL == 0)
        {
            printf("It. %u, loss = %.4f\n", i+1, loss);
            result.losses[i] = loss;
        }
        
        // Update theta
        backward(&x_batch, &y_batch, 
                 &(result.theta_sol), learning_rate, grad_fn);
    }

    if (result.converged)
        printf("Converged in %u iterations.\n", i+1);

    // Calculate predicted bias
    // bias = y_offset - x_offset theta
    x_offset_theta_prod = mat_mul(&x_offset, false, 
                                &(result.theta_sol), false);
    mat_sub(&y_offset, &x_offset_theta_prod);
    bias_full = stats_mean(&y_offset, 2);
    result.bias = bias_full.data[0];
    
    // Destroy local matrices
    mat_destroy(&x_copy);
    mat_destroy(&y_copy);
    mat_destroy(&x_offset);
    mat_destroy(&y_offset);
    mat_destroy(&x_batch);
    mat_destroy(&y_batch);
    mat_destroy(&x_offset_theta_prod);
    mat_destroy(&y_pred);
    mat_destroy(&bias_full);
    intmat_destroy(&idxs);

    // Truncate loss array
    if (i>LOSS_INTERVAL)
        result.losses = (double *)realloc(result.losses, 
                        ((i+1)/LOSS_INTERVAL) * sizeof(double));

    return result;
}
