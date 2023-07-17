# SparseModule
A module for Numba accelerated calculations using sparse matrices in Python. 

Newer Scipy implementations for calculations using sparse matrices are moving towards an object oriented form, which is harder to use with different acceleration methods available for Python. Most of this code is based directly off of Scipy, requires the use of Numpy arrays, and already includes Numba acceleration (although currently there is no parallelization).
