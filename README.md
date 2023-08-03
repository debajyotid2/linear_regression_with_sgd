# Linear Regression using Stochastic Gradient Descent

This repository is a demonstration of linear regression solved using low level matrix and linear algebra operations in C and NumPy in Python. 

## Dependencies

### C 

The C source code uses BLAS (Basic Linear Algebra Subsystems) libraries for matrix and vector operations. The implementation specifically used is OpenBLAS (https://github.com/xianyi/OpenBLAS.git). Please refer to the installation instructions for OpenBLAS in its GitHub repository. The library `matrix.h` supports elementary algebraic oprations for two-dimensional matrices. Data types currently supported are `double` (for double-precision floating point) and `int` (for integers).

To run a multivariate linear regression problem with the default settings and stochastic gradient descent solver, run

```
cd c
make
./run
```

Hyperparameters can be customized from the command line. For more details, run `./run --help`.

Warning: Code has only been tested on Fedora 38 Linux. 

### Python

The Python source code uses only matrix operations in NumPy for implementing gradient descent and stochastic gradient descent. The only dependencies are `numpy` and `matplotlib`.
