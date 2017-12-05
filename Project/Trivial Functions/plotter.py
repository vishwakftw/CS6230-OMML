import sys
import numpy as np
import functions as f
import gradients as g
import optimizers as o
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

# cur_func is one of B1, B2, B3, RB and BL
cur_func = sys.argv[1]
color_dict = {'GD': 'black', 'CM': 'blue', 'Adagrad': 'gold', 'Adam': 'green', 'RMSprop': 'red'}

# Plot the contour of the function
if cur_func == 'B1' or cur_func == 'B2' or cur_func == 'B3':
    X, Y = np.meshgrid(np.arange(-25, 25, 0.5), np.arange(-25, 25, 0.5))
    plt.contour(X, Y, f._func_dicts[cur_func]([X, Y]), levels=np.linspace(np.amin(f._func_dicts[cur_func]([X, Y])), np.amax(f._func_dicts[cur_func]([X, Y])), 50))
if cur_func == 'RB':
    X, Y = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-6, 6, 0.1))
    plt.contour(X, Y, f._func_dicts[cur_func]([X, Y]), levels=np.linspace(np.amin(f._func_dicts[cur_func]([X, Y])), np.amax(f._func_dicts[cur_func]([X, Y])), 50))
if cur_func == 'BL':
    X, Y = np.meshgrid(np.arange(-4.5, 4.5, 0.05), np.arange(-4.5, 4.5, 0.05))
    plt.contour(X, Y, f._func_dicts[cur_func]([X, Y]), levels=np.logspace(0, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)

# First get the iterates
opt_params = np.genfromtxt('best_values.txt', dtype=str)
opt_params = opt_params[opt_params[:,0] == cur_func]
title = ''
for i in range(0, opt_params.shape[0]):
    iterates = np.genfromtxt('{0}/{1}/iter_{2}_{0}.txt'.format(cur_func, opt_params[i,1], opt_params[i,2]))
    plt.plot(iterates[:,0], iterates[:,1], color=color_dict[opt_params[i,1]], label=opt_params[i,1], linewidth=2.5)
    title = title + '{0}: {1}   '.format(opt_params[i,1], iterates[-1])
plt.title(title)
plt.legend(loc='upper right')
plt.show()
