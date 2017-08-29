import numpy as np

def quad(x, y):
	return 1.125*x**2 + 0.5*x*y + 0.75*y**2 + 2*x + 2*y
	
def log_reg(x, y):
	return 0.5*(x**2 + y**2) + 50*np.log(1 + np.exp(-0.5*y)) + 50*np.log(1 + np.exp(0.2*x))
	
def himmelblaus(x, y):
	return 0.1*(x**2 + y - 11)**2 + 0.1*(x + y**2 - 7)**2
	
def rosenbrock(x, y):
	return 0.002*(1 - x)**2 + 0.2*(y - x**2)**2
	
def grad_quad(x, y):
	dx	= 2.25*x + 0.5*y + 2
	dy	= 1.5*y + 0.5*x + 2
	return np.array([dx, dy])
	
def grad_log_reg(x, y):
	dx1	= x
	dx2	= 0
	dx3	= 10*np.exp(0.2*x)/(1 + np.exp(0.2*x))
	dy1	= y
	dy2	= -25*np.exp(-0.5*y)/(1 + np.exp(-0.5*y))
	dy3	= 0
	dx	= dx1 + dx2 + dx3
	dy	= dy1 + dy2 + dy3
	return np.array([dx, dy])
	
def grad_himmelblaus(x, y):
	dt1	= 0.2*(x**2 + y - 11)
	dt2	= 0.2*(x + y**2 - 7)
	dt1x	= 2*x
	dt2x	= 1
	dt1y	= 1
	dt2y	= 2*y
	dx	= dt1*dt1x + dt2*dt2x
	dy	= dt1*dt1y + dt2*dt2y
	return np.array([dx, dy])
	
def grad_rosenbrock(x, y):
	dt1	= 0.004*(1 - x)
	dt2	= 0.4*(y - x**2)
	dt1x	= -1
	dt2x	= -2*x
	dt1y	= 0
	dt2y	= 1
	dx	= dt1*dt1x + dt2*dt2x
	dy	= dt1*dt1y + dt2*dt2y
	return np.array([dx, dy])
