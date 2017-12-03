import sys
import os, errno
import numpy as np
import functions as f
import gradients as g
import optimizers as o
from argparse import ArgumentParser as AP

def give_init(function_name):
    if function_name == 'B1':
        return np.array([-75, 50])
    elif function_name == 'B2':
        return np.array([50, -50])
    elif function_name == 'B3':
        return np.array([-50, 50])
    elif function_name == 'BL':
        return np.array([1, 2])
    elif function_name == 'RB':
        return np.array([-2.1, 2.1])
    elif function_name == 'ST':
        return np.array([0, 0])

p = AP()
p.add_argument('--optim', required=True, help='Optimizers: GD | CM | Adam | Adagrad | Adadelta | RMSprop')
p.add_argument('--fn', required=True, help='Function name: B1 | B2 | B3 | BL | RB | ST')
p.add_argument('--iter', default=1000, type=int, help='Number of iterations to run for')
p = p.parse_args()


newDir = os.path.join(p.fn, p.optim)
try:
    os.makedirs(newDir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if p.optim == 'GD':
    learning_rates = np.around(np.logspace(-4, -2.5, 12), decimals=5)
    for lr in learning_rates:
        i = 0
        X = give_init(p.fn)
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X = o._opt_dicts[p.optim](i, X, G, lr)
        log_file.close()

elif p.optim == 'CM':
    learning_rates = np.around(np.logspace(-4, -2.5, 12), decimals=5)
    for lr in learning_rates:
        v = 0
        m = 0.925
        i = 0
        X = give_init(p.fn)
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, v = o._opt_dicts[p.optim](i, X, G, v, lr, m)
        log_file.close()

elif p.optim == 'Adam':
    learning_rates = np.around(np.logspace(-4, -1.5, 18), decimals=5)
    for lr in learning_rates:
        fm = 0
        sm = 0
        i = 0
        X = give_init(p.fn)
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, fm, sm = o._opt_dicts[p.optim](i, X, G, lr, fm, sm)
        log_file.close()

elif p.optim == 'Adagrad':
    learning_rates = np.around(np.logspace(-4, 0, 30), decimals=5)
    for lr in learning_rates:
        ssg = 0
        i = 0
        X = give_init(p.fn)
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, ssg = o._opt_dicts[p.optim](i, X, G, ssg, lr)
        log_file.close()

elif p.optim == 'Adadelta':
    learning_rates = np.around(np.logspace(-4, 0, 30), decimals=5)
    for lr in learning_rates:
        eg = 0
        edx = 0
        i = 0
        X = give_init(p.fn)
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, eg, edx = o._opt_dicts[p.optim](i, X, G, eg, edx, lr)
        log_file.close()

elif p.optim == 'RMSprop':
    learning_rates = np.around(np.logspace(-4, -1.5, 18), decimals=5)
    for lr in learning_rates:
        eg = 0
        i = 0
        X = give_init(p.fn)
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, eg = o._opt_dicts[p.optim](i, X, G, eg, lr)
        log_file.close()
