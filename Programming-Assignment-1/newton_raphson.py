import numpy as np

def f(z, a):
	return z + 50*a*np.exp(a*z)/(1 + np.exp(a*z))
	
def df(z, a):
	return 1 + 50*a*(a*np.exp(a*z)/(1 + np.exp(a*z))**2)
	
def fxy(x, y):
	return 0.5*(x**2 + y**2) + 50*(np.log(1 + np.exp(-0.5*y))) + 50*(np.log(1 + np.exp(0.2*x)))

x	= -1
for i in range(0, 50):
	x	= x - f(x, 0.2)/df(x, 0.2)
	
y	= 1
for i in range(0, 50):
	y	= y - f(y, -0.5)/df(y, -0.5)
	
print('{0} at {1}'.format(f(x, 0.2), x))
print('{0} at {1}'.format(f(y, -0.5), y))

print(fxy(x, y))
