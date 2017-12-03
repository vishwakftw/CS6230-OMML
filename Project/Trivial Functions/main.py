import sys
import numpy as np
import functions as f
import gradients as g
import optimizers as o
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--optim', required=True, help='Optimizers: GD | CM | Adam | Adagrad | Adadelta | RMSprop')
p.add_argument('--lr', type=float, required=True, help='Learning rate')
p.add_argument('--fn', required=True, help='Function name: B1 | B2 | B3 | BL | RB | ST')
p.add_argument('--iter', default=250, type=int, help='Number of iterations to run for')
p = p.parse_args()

i = 0
X = np.full(2, 2)
log_file = open('log_{0}_{1}_{2}.txt'.format(p.optim, p.lr, p.fn), 'w')
while i < p.iter:
    log_file.write('{0}\t{1}\n'.format(i, f._func_dicts[p.fn](X)))
    G = g._grad_dicts[p.fn](X)
    i, X = o._opt_dicts[p.optim](i, X, G, p.lr)
log_file.close()