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
        Returns gradient of Bohachevsky Function 1 at a point
    """
    x = X[0]
    y = X[1]
    dx = 2*x + 3.0*np.pi*0.3*np.sin(3.0*np.pi*x)
    dy = 4*y + 4.0*np.pi*0.4*np.sin(4.0*np.pi*y)
    return np.array([dx, dy])

def B2(X):
    """
        Returns gradient of Bohachevsky Function 2 at a point
    """
    x = X[0]
    y = X[1]
    dx = 2*x + 3.0*np.pi*0.3*np.sin(3.0*np.pi*x)*np.cos(4.0*np.pi*y)
    dy = 4*y + 4.0*np.pi*0.4*np.sin(4.0*np.pi*y)*np.cos(3.0*np.pi*x)
    return np.array([dx, dy])

def B3(X):
    """
        Returns gradient of Bohachevsky Function 3 at a point
    """
    x = X[0]
    y = X[1]
    dx = 2*x + 3.0*np.pi*0.3*np.sin(3.0*np.pi*x + 4.0*np.pi*y)
    dy = 4*y + 4.0*np.pi*0.4*np.sin(3.0*np.pi*x + 4.0*np.pi*y)
    return np.array([dx, dy])

def RB(X):
    """
        Returns gradient of Rosenbrock's function at a point
    """
    x = X[0]
    y = X[1]
    dx = -100*(y - x**2)*x + 2*(x - 1)
    dy = 50*(y - x**2)
    return np.array([dx, dy])

def BL(X):
    """
        Returns gradient of Beale's function at a point
    """
    x = X[0]
    y = X[1]
    cmn1    = 2*(1.5 - x + x*y)
    cmn2    = 2*(2.25 - x + x*y**2)
    cmn3    = 2*(2.625 - x + x*y**3)
    tmx1    = -1 + y
    tmx2    = -1 + y**2
    tmx3    = -1 + y**3
    tmy1    = x
    tmy2    = 2*x*y
    tmy3    = 3*x*y**2
    dx  = cmn1*tmx1 + cmn2*tmx2 + cmn3*tmx3
    dy  = cmn1*tmy1 + cmn2*tmy2 + cmn3*tmy3
    return np.array([dx, dy])

def ST(X):
    """
        Returns gradient of Styblinsky-Tang's function at a point
    """
    x = X[0]
    y = X[1]
    dx = 4*x**3 - 32*x + 5
    dy = 4*y**3 - 32*y + 5
    return np.array([dx, dy])

_grad_dicts = {'B1': B1, 'B2': B2, 'B3': B3, 'RB': RB, 'BL': BL, 'ST': ST}
