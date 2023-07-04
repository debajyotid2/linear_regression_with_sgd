// Miscellaneous statistical functions

#ifndef _STATS_H_
#define _STATS_H_

#include "matrix.h"

Matrix stats_mean(Matrix* mat, unsigned int dimension);
double stats_mae(Matrix* y_true, Matrix* y_pred);
double stats_r2(Matrix* y_true, Matrix* y_pred);

#endif // _STATS_H_
