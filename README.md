# Linear Regression using Stochastic Gradient Descent

This repository is to build my own understanding of low level matrix and linear algebra operations in C and NumPy in Python. 

## Dependencies

### C 

The C source code uses BLAS (Basic Linear Algebra Subsystems) libraries for matrix and vector operations. The implementation specifically used is OpenBLAS (https://github.com/xianyi/OpenBLAS.git). Refer to the installation instructions for OpenBLAS in its GitHub repository. Then to run the regression with the default settings, run

```
cd c
make
./run
```

Hyperparameters can be customized from the command line. For more details, run `./run --help`.

### Python

The Python source code uses only matrix operations in NumPy for implementing gradient descent and stochastic gradient descent. The only dependencies are `numpy` and `matplotlib`.
