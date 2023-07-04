/* Library for reading, writing and manipulating matrices in C. */

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdio.h>
#include <stdbool.h>

// Matrix for double precision data
typedef struct
{
    unsigned int nrows, ncols;
    double *data;
} Matrix;

// Matrix for integer data
typedef struct
{
    unsigned int nrows, ncols;
    int *data;
} IntMatrix;

// Functions for integer matrices
IntMatrix intmat_create(int nrow, int ncol);
IntMatrix intmat_copy(IntMatrix* mat);
void intmat_copy_inplace(IntMatrix* mat, IntMatrix* copy);
IntMatrix intmat_range(int low, int high, unsigned int step,
                        unsigned int dimension);
void intmat_print(const IntMatrix* matrix);
void intmat_fill(IntMatrix* matrix, int value);
void intmat_fill_random(IntMatrix* matrix, int low, int high, 
                    bool replace, unsigned int seed);
void intmat_scale(IntMatrix* mat, int fac);
void intmat_add_scalar(IntMatrix* mat, int scalar);
void intmat_add(IntMatrix* mat_a, IntMatrix* mat_b);
void intmat_sub(IntMatrix* mat_a, IntMatrix* mat_b);
IntMatrix intmat_mul(IntMatrix* mat_a, bool transpose_a, 
               IntMatrix* mat_b, bool transpose_b);
void intmat_mul_inplace(IntMatrix* mat_a, bool transpose_a, IntMatrix* mat_b,
                     bool transpose_b, IntMatrix* result);
IntMatrix intmat_repeat(IntMatrix* vec, unsigned int dimension, 
                        unsigned int repeats);
void intmat_vec_add(IntMatrix* mat, IntMatrix* vec);
void intmat_vec_sub(IntMatrix* mat, IntMatrix* vec);
void intmat_gather(IntMatrix* from, IntMatrix* to, IntMatrix* indices,
                   unsigned int dimension);
void intmat_destroy(IntMatrix* matrix);


// Functions for double matrices
Matrix mat_create(int nrow, int ncol);
Matrix mat_copy(Matrix* mat);
void mat_copy_inplace(Matrix* mat, Matrix* copy);
void mat_print(const Matrix* matrix);
Matrix mat_range(double low, double high, double step,
                  unsigned int dimension);
void mat_fill(Matrix* matrix, double value);
void mat_fill_random(Matrix* matrix, unsigned int seed);
void mat_fill_random_gaussian(Matrix* matrix, Matrix* means, 
                            Matrix* stds, unsigned int seed);
void mat_scale(Matrix* mat, double fac);
double mat_abs_sum(Matrix* mat);
double mat_norm(Matrix* mat);
void mat_add_scalar(Matrix* mat, double scalar);
void mat_add(Matrix* mat_a, Matrix* mat_b);
void mat_sub(Matrix* mat_a, Matrix* mat_b);
Matrix mat_mul(Matrix* mat_a, bool transpose_a, 
               Matrix* mat_b, bool transpose_b);
void mat_mul_inplace(Matrix* mat_a, bool transpose_a, Matrix* mat_b,
                     bool transpose_b, Matrix* result);
Matrix mat_repeat(Matrix* vec, unsigned int dimension, unsigned int repeats);
void mat_vec_add(Matrix* mat, Matrix* vec);
void mat_vec_sub(Matrix* mat, Matrix* vec);
void mat_gather(Matrix* from, Matrix* to, IntMatrix* indices,
                unsigned int dimension);
void mat_destroy(Matrix* matrix);

#endif // _MATRIX_H_
