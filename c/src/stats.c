// Miscellaneous statistical functions.

#include <stdlib.h>
#include "matrix.h"
#include "stats.h"

// Mean of a two-dimensional matrix along
// the specified dimension - 0 is columns and 1 is rows.
// Any other number specified gives the mean across all dimensions.
Matrix stats_mean(Matrix* mat, unsigned int dimension)
{
    Matrix mean;
    
    // Reduce along rows
    if (dimension==0)
    {
        mean = mat_create(1, mat->ncols);
        mat_fill(&mean, 0.0);

        for (size_t i=0; i<mat->nrows*mat->ncols; ++i)
            mean.data[i%mat->ncols] += mat->data[i];
        mat_scale(&mean, 1.0/(double)mat->nrows);
    }
    // Reduce along columns
    else if (dimension==1)
    {
        mean = mat_create(mat->nrows, 1);
        mat_fill(&mean, 0.0);

        for (size_t i=0; i<mat->nrows*mat->ncols; ++i)
            mean.data[i%mat->nrows] += mat->data[i];
        mat_scale(&mean, 1.0/(double)mat->ncols);
    }
    // Reduce along all axes
    else
    {
        mean = mat_create(1, 1);
        mat_fill(&mean, 0.0);

        for (size_t i=0; i<mat->nrows*mat->ncols; ++i)
            mean.data[0] += mat->data[i];
        mean.data[0] /= (double)(mat->nrows*mat->ncols);
    }

    return mean;
}

// Mean absolute error.
// MAE = \sum_i (y_true_i - y_pred_i)
double stats_mae(Matrix* y_true, Matrix* y_pred)
{
    if (y_true==NULL || y_pred==NULL)
    {
        perror("ERROR: Null pointers in array arguments.");
        return 0.0;
    }
    if (y_true->nrows!=y_pred->nrows ||
            y_true->ncols!=1 || y_pred->ncols!=1)
    {
        perror("ERROR: Arrays do not match expected dimensions.");
        return 0.0;
    }
    
    double mae = 0.0;
    Matrix diff = mat_copy(y_true);
    
    mat_sub(&diff, y_pred);
    mae = mat_abs_sum(&diff);

    mat_destroy(&diff);

    return mae / (y_true->nrows*y_true->ncols);
}

// Coefficient of determination
// R^2 = 1 - SS_{res}/SS_{tot}
// where SS_{res} = \sum_i (y_true_i - y_pred_i)^2
//       SS_{tot} = \sum_i (y_true_i - y_mean)
double stats_r2(Matrix* y_true, Matrix* y_pred)
{
    if (y_true==NULL || y_pred==NULL)
    {
        perror("ERROR: Null pointers in array arguments.");
        return 0.0;
    }
    if (y_true->nrows!=y_pred->nrows ||
            y_true->ncols!=1 || y_pred->ncols!=1)
    {
        perror("ERROR: Arrays do not match expected dimensions.");
        return 0.0;
    }
    
    double ss_res, ss_tot;
    Matrix y_mean, diff, ss_tot_vec;
   
    diff = mat_copy(y_true);
    mat_sub(&diff, y_pred);
    ss_res = mat_norm(&diff);
    ss_res = ss_res*ss_res;

    y_mean = stats_mean(y_true, 0);
    ss_tot_vec = mat_copy(y_true);
    mat_add_scalar(&ss_tot_vec, -1 * y_mean.data[0]);
    ss_tot = mat_norm(&ss_tot_vec);
    ss_tot = ss_tot * ss_tot;

    mat_destroy(&y_mean);
    mat_destroy(&diff);
    mat_destroy(&ss_tot_vec);

    return 1.0 - ss_res/ss_tot;
}

