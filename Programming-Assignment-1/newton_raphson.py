# Newton-Raphson to find the roots of equation
# g_a(x) = x + 50*a*exp(ax)/(1 + exp(ax))
# This happens to be the form of the partial derivatives of the Ridge Regularized Logistic Regression function
import numpy as np

def g(z, a):
	return z + 50*a*np.exp(a*z)/(1 + np.exp(a*z))

def dg(z, a):
	return 1 + 50*a*(a*np.exp(a*z)/(1 + np.exp(a*z))**2)

def f_ll(x, y):
	return 0.5*(x**2 + y**2) + 50*(np.log(1 + np.exp(-0.5*y))) + 50*(np.log(1 + np.exp(0.2*x)))

x	= -1
for i in range(0, 50):
	x	= x - g(x, 0.2)/dg(x, 0.2)	

y	= 1
for i in range(0, 50):
	y	= y - g(y, -0.5)/dg(y, -0.5)
	
print('Partial Derivative w.r.t x => {0} at {1} \
	\nPartial Derivative w.r.t y => {2} at {3} \
	\nFunction Value is: {4}'.format(round(g(x, 0.2), 10), round(x, 4), round(g(y, -0.5), 10), round(y, 4), round(f_ll(x, y), 4)))
