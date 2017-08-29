import numpy as np
from functions import *
from matplotlib import pyplot as plt

def update(cur_point, lr, func):
	gradients = {'quad'		: grad_quad,
		     'log_reg'		: grad_log_reg,
		     'himmelblaus'	: grad_himmelblaus,
		     'rosenbrock'	: grad_rosenbrock
		    }
	updt	= cur_point - lr*gradients[func](*cur_point)
	return updt
