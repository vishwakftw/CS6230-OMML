import h5py as h
import cvxpy as C
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--C1', required=True, type=float, help='C1 parameter')
p.add_argument('--C2', required=True, type=float, help='C2 parameter')
opt = p.parse_args()

# Load the dataset
f = h.File('toy.hdf5', 'r')
X = np.array(list(f['X'])).T
y = np.array(list(f['y'])).T
print('Dataset loaded: X.shape = {0}'.format(X.shape))

# Declare the variables to optimize over
m, d = X.shape[0], X.shape[1]
alpha = C.Variable(m)

# Declare the objective function
# First calculate X(tilde)X(tilde)^{T}
constant_matrix = np.zeros((m, m))
for i in range(0, m):
    for j in range(0, m):
        constant_matrix[i,j] = y[i]*y[j]*np.dot(X[i], X[j])

obj_1 = C.sum_entries(alpha)
obj_2 = C.sum_entries(C.quad_form(alpha, constant_matrix))*0.5
objective = obj_1 - obj_2

# Declare the constraints to enforce
constraints = [C.sum_entries(alpha.T*y) == 0, alpha >= 0]
for i in range(0, m):
    if y[i] == 1:    
        constraints += [alpha[i] <= opt.C1]
    else:
        constraints += [alpha[i] <= opt.C2]

# Define the problem
problem = C.Problem(C.Maximize(objective), constraints)

# Solve the problem
problem.solve(solver=C.CVXOPT)

print("Problem exited with status: {0} and value attained: {1}".format(problem.status, round(problem.value, 5)))
