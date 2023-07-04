// Stochastic gradient descent

#ifndef _SGD_H_
#define _SGD_H_

#include <stdbool.h>
#include "matrix.h"

// Loss functions and gradient functions
typedef double (*loss_fn_type)(Matrix*, Matrix*);
typedef Matrix (*grad_fn_type)(Matrix*, Matrix*, Matrix*);

// Result struct for stochastic gradient descent
typedef struct
{
    bool converged;
    double bias;
    unsigned int n_iter;
    double* losses;
    Matrix theta_sol;
} SGDResult;

void init_sgdresult(SGDResult* result,
                    unsigned int n_iter,
                    unsigned int n_features,
                    unsigned int seed);
void destroy_sgdresult(SGDResult* result);
void forward(Matrix* x, Matrix* theta, Matrix* y_pred);
void backward(Matrix* x, Matrix* y, 
              Matrix* theta, double eta,
              grad_fn_type grad_fn);
SGDResult gradient_descent(
            Matrix* x, Matrix* y,
            double learning_rate,
            loss_fn_type loss_fn,
            grad_fn_type grad_fn,
            unsigned int n_iter,
            double tol,
            unsigned int seed);
SGDResult stochastic_gradient_descent(
            Matrix* x, Matrix* y,
            unsigned int batch_size,
            double learning_rate, 
            loss_fn_type loss_fn,
            grad_fn_type grad_fn,
            unsigned int n_iter,
            double tol,
            unsigned int seed);

#endif // _SGD_H_
