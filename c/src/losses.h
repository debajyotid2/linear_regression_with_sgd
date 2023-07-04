#ifndef _LOSSES_H_
#define _LOSSES_H_

#include "matrix.h"

double l2_loss(Matrix* y_true, Matrix* y_pred);
Matrix l2_gradient(Matrix* x, Matrix* y, Matrix* theta);

#endif // _LOSSES_H_
