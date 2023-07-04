#ifndef _HELPERS_H_
#define _HELPERS_H_

#include <stdlib.h>

void make_regression_dataset(Matrix* x, Matrix* y,
                             double bias, double noise_intensity,
                             unsigned int seed);
void split_into_train_test(Matrix* x, Matrix* y,
                           Matrix* x_train, Matrix* y_train,
                           Matrix* x_test, Matrix* y_test,
                           unsigned int seed);

#endif // _HELPERS_H_
