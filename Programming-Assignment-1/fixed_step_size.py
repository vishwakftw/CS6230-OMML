import numpy as np
from matplotlib import pyplot as plt

def quad(x, y):
	return 1.125*x**2 + 0.5*x*y + 0.75*y**2 + 2*x + 2*y
	
def log_reg(x, y):
	return 0.5*(x**2 + y**2) + 50*np.log(1 + np.exp(-0.5*y)) + 50*(1 + np.exp(0.2*x))
	
def himmelblaus(x, y):
	return 0.1*(x**2 + y - 11)**2 + 0.1*(x + y**2 - 7)**2
	
def rosenbrock(x, y):
	return 0.002*(1 - x)**2 + 0.2*(y*x**2)**2
