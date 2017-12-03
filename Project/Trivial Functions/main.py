import sys
import numpy as np
import functions as f
import gradients as g
import optimizers as o
from argparse import ArgumentParser as AP
import os, errno

p = AP()
p.add_argument('--optim', required=True, help='Optimizers: GD | CM | Adam | Adagrad | Adadelta | RMSprop')
p.add_argument('--fn', required=True, help='Function name: B1 | B2 | B3 | BL | RB | ST')
p.add_argument('--iter', default=250, type=int, help='Number of iterations to run for')
p = p.parse_args()

i = 0
X = np.full(2, 2)

learning_rates = np.logspace(-4, -2, 12)

newDir = p.fn + '/' + p.optim
try:
    os.makedirs(newDir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


if p.optim == 'GD':
    for lr in learning_rates:
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            G = g._grad_dicts[p.fn](X)
            i, X = o._opt_dicts[p.optim](i, X, G, lr)
        log_file.close()

elif p.optim == 'CM':
    v = 0
    m = 0.95
    for lr in learning_rates:
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            G = g._grad_dicts[p.fn](X)
            i, X, v = o._opt_dicts[p.optim](i, X, G, v, lr, m)
        log_file.close()

elif p.optim == 'Adam':
    fm = 0
    sm = 0
    for lr in learning_rates:
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, fm, sm = o._opt_dicts[p.optim](i, X, G, lr, fm, sm)
        log_file.close()

elif p.optim == 'Adagrad':
    ssg = 0
    for lr in learning_rates:
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, ssg = o._opt_dicts[p.optim](i, X, G, ssg, lr)
        log_file.close()

elif p.optim == 'Adadelta':
    eg = 0
    edx = 0
    for lr in learning_rates:
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, eg, edx = o._opt_dicts[p.optim](i, X, G, eg, edx, lr)
        log_file.close()

elif p.optim == 'RMSprop':
    eg = 0
    for lr in learning_rates:
        log_file = open('{0}/log_{1}_{2}.txt'.format(newDir, lr, p.fn), 'w')
        while i < p.iter:
            log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
            G = g._grad_dicts[p.fn](X)
            i, X, eg = o._opt_dicts[p.optim](i, X, G, egC, lr)
        log_file.close()




