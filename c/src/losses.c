// Loss functions in linear regression

#include <stdlib.h>
#include "matrix.h"
#include "losses.h"

// l2 loss = ||y_true - y_pred||^2/N
// where y_true is ground truth,
// y_pred is predicted values and N is the 
// number of observations.
double l2_loss(Matrix* y_true, Matrix* y_pred)
{
    Matrix diff;
    double loss;
    
    // diff = y_true - y_pred
    diff = mat_copy(y_true);
    mat_sub(&diff, y_pred);

    // loss = ||diff||^2
    loss = mat_norm(&diff);
    loss = loss*loss;

    mat_destroy(&diff);

    return loss / y_pred->nrows;
}
 
// Gradient of the L2 loss with respect to theta (coefficients)
// given by gradient = X.T (X theta - y)
Matrix l2_gradient(Matrix* x, Matrix* y, Matrix* theta)
{
    Matrix grad, x_theta_prod;
    
    // Calculate gradient
    x_theta_prod = mat_mul(x, false, theta, false);
    mat_sub(&x_theta_prod, y);
    grad = mat_mul(x, true, &x_theta_prod, false);

    mat_destroy(&x_theta_prod);
 
    return grad;
}
