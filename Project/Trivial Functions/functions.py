import numpy as np

# Index of references
# B1 -> Bohachevsky Function 1
# B2 -> Bohachevsky Function 2
# B3 -> Bohachevsky Function 3
# RB -> Rosenbrock Function
# BL -> Beale's Function
# ST -> Styblinsky-Tang's Function

def B1(X):
    """
        Returns Bohachevsky function 1 at a point
    """
    x = X[0]
    y = X[1]
    return x**2 + 2.0*(y**2) - 0.3*np.cos(3.0*np.pi*x) - 0.4*np.cos(4.0*np.pi*y) + 0.7

def B2(X):
    """
        Returns Bohachevsky function 2 at a point
    """
    x = X[0]
    y = X[1]
    return x**2 + 2.0*(y**2) - 0.3*np.cos(3.0*np.pi*x)*np.cos(4.0*np.pi*y) + 0.3

def B3(X):
    """
        Returns Bohachevsky function 3 at a point
    """
    x = X[0]
    y = X[1]
    return x**2 + 2.0*(y**2) - 0.3*np.cos(3.0*np.pi*x + 4.0*np.pi*y) + 0.3

def RB(X):
    """
        Returns Rosenbrock's function at a point
    """
    x = X[0]
    y = X[1]
    return 25*(y - x**2)**2 + (1 - x)**2

def BL(X):
    """
        Returns Beale's function at a point
    """
    x = X[0]
    y = X[1]
    term1 = (1.5 - x + x*y)**2
    term2 = (2.25 - x + x*y**2)**2
    term3 = (2.625 - x + x*y**3)**2
    return term1 + term2 + term3

_func_dicts = {'B1': B1, 'B2': B2, 'B3': B3, 'RB': RB, 'BL': BL}
