// Helper functions

#include <stdlib.h>
#include <stdbool.h>
#include "matrix.h"
#include "helpers.h"

// Make a dataset for solving a linear regression problem.
// The data is generated as y = x * coeff + noise + bias
void make_regression_dataset(Matrix* x, 
                             Matrix* y, 
                             double bias,
                             double noise_intensity,
                             unsigned int seed)
{
    Matrix prod;
    Matrix noise_mean = mat_create(1, 1);
    Matrix noise_std = mat_create(1, 1);
    Matrix means = mat_create(x->ncols, 1);
    Matrix stds = mat_create(x->ncols, 1);
    Matrix coeff = mat_create(x->ncols, 1);
    Matrix bias_vec = mat_create(x->nrows, 1);
    Matrix noise_vec = mat_create(x->nrows, 1);

    // Ensure that y has same number of rows as x
    if (x->nrows != y->nrows || y->ncols != 1)
    {
        perror("ERROR: Dimensions of y must be M x 1 (if dim(X) = M x N)");
        mat_destroy(&bias_vec);
        mat_destroy(&noise_vec);
        mat_destroy(&coeff);
        return;
    }

    // Generate x matrix
    mat_fill_random(&means, seed);
    mat_scale(&means, 10.0);
    mat_fill_random(&stds, seed);
    mat_scale(&stds, 10.0);
    mat_fill_random_gaussian(x, &means, &stds, seed);

    // Generate bias array
    mat_fill(&bias_vec, bias);

    // Generate noise array
    mat_fill(&noise_mean, 0.0);
    mat_fill(&noise_std, 1.0);
    mat_fill_random_gaussian(&noise_vec, &noise_mean, &noise_std, seed);
    mat_scale(&noise_vec, noise_intensity);

    // Generate coefficients to be between -25.0 and 25.0
    mat_fill_random(&coeff, seed);
    mat_scale(&coeff, 50.0);
    mat_add_scalar(&coeff, -25.0);

    // Generate y = x coeff + bias + noise
    prod = mat_mul(x, false, &coeff, false);
    mat_add(&prod, &bias_vec);
    mat_add(&prod, &noise_vec);

    mat_copy_inplace(&prod, y);

    // Delete means and stds
    mat_destroy(&means);
    mat_destroy(&stds);
    mat_destroy(&noise_mean);
    mat_destroy(&noise_std);

    // Delete coefficients, noise_vec, prod and bias_vec
    mat_destroy(&noise_vec);
    mat_destroy(&bias_vec);
    mat_destroy(&prod);
    mat_destroy(&coeff);
}

// Split x_train and y_train into training and test sets
void split_into_train_test(Matrix* x, Matrix* y,
                           Matrix* x_train, Matrix* y_train,
                           Matrix* x_test, Matrix* y_test,
                           unsigned int seed)
{
    if (x_train==NULL || y_train==NULL || x_test==NULL || y_test==NULL \
            || x==NULL || y==NULL)
    {
        perror("ERROR: Null value(s) detected in argument arrays.");
        return;
    }
    if (x_train->ncols!=x_test->ncols || y_train->ncols!=1 || y_test->ncols!=1)
    {
        perror("ERROR: Mismatch of n_features between training and test sets.");
        mat_destroy(x_train);
        return;
    }
    if (x_test->nrows>=x_train->nrows || y_test->nrows>=y_train->nrows)
    {
        perror("ERROR: Number of rows of test data matrix must be less than train matrix.");
        mat_destroy(x_train);
        return;
    }
    if (x_train->nrows+x_test->nrows!=x->nrows || y_train->nrows+y_test->nrows!=y->nrows)
    {
        perror("ERROR: Number of rows in train and test matrices do not add up to total rows in data.");
        mat_destroy(x_train);
        return;
    }
    
    IntMatrix rand_idxs = intmat_create(x->nrows, 1);
    IntMatrix train_idxs = intmat_create(x_train->nrows, 1);
    IntMatrix test_idxs = intmat_create(x_test->nrows, 1);
    IntMatrix range_idxs = intmat_range(0, x_test->nrows, 1, 0);
    
    // Fill random test indices
    intmat_fill_random(&rand_idxs, 0, x->nrows, false, seed);
    intmat_gather(&rand_idxs, &test_idxs, &range_idxs, 0);
    
    // Fill random train indices
    intmat_destroy(&range_idxs);
    range_idxs = intmat_range(x_test->nrows, x->nrows, 1, 0);
    intmat_gather(&rand_idxs, &train_idxs, &range_idxs, 0);
    
    // Fill x_test and y_test
    mat_gather(x, x_test, &test_idxs, 0);
    mat_gather(y, y_test, &test_idxs, 0);

    // Fill x_train and y_train
    mat_gather(x, x_train, &train_idxs, 0);
    mat_gather(y, y_train, &train_idxs, 0);

    intmat_destroy(&rand_idxs);
    intmat_destroy(&train_idxs);
    intmat_destroy(&test_idxs);
    intmat_destroy(&range_idxs);
}
