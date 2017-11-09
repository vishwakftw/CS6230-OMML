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
beta = C.Variable(d)
beta_0 = C.Variable(1)
slackvar = C.Variable(m)

# Declare the objective function
objective = (C.sum_squares(beta))*0.5
for i in range(0, m):
    if y[i] == 1:
        objective += slackvar[i]*opt.C1
    else:
        objective += slackvar[i]*opt.C2

# Declare the constraints to enforce
constraints = [slackvar >= 0]
for i in range(0, m):
    constraints += [y[i]*(beta.T*X[i] + beta_0) >= 1 - slackvar[i]]

# Define the problem
problem = C.Problem(C.Minimize(objective), constraints)

# Solve the problem
problem.solve(solver=C.CVXOPT)

print("Problem exited with status: {0} and value attained: {1}".format(problem.status, problem.value))

# Plotting section
label_flag = [False, False]
for i in range(0, m):
    if y[i] == 1:
        if label_flag[0] == False:        
            plt.plot(X[i,0], X[i,1], 'go', label='+1')
            label_flag[0] = True
        else:
            plt.plot(X[i,0], X[i,1], 'go')
    else:
        if label_flag[1] == False:        
            plt.plot(X[i,0], X[i,1], 'bo', label='-1')
            label_flag[1] = True
        else:
            plt.plot(X[i,0], X[i,1], 'bo')

x = np.arange(np.amin(X[:,0]), np.amax(X[:,0]), 0.001)
beta = (beta.value).tolist()
beta_0 = beta_0.value
y = -beta_0/beta[1] - beta[0]*x/beta[1]
plt.plot(x, y, 'r--')

plt.legend(loc='upper right', numpoints=1)
plt.show()
