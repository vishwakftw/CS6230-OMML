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

print("Problem exited with status: {0} and value attained: {1}".format(problem.status, round(problem.value, 5)))

# Plotting section
label_flag = [False, False]
for i in range(0, m):
    if y[i] == 1:
        point_type = 'ro' if (slackvar[i].value) > 1e-07 else 'go'

        if label_flag[0] == False and point_type == 'go':
            plt.plot(X[i,0], X[i,1], point_type, label='+1')
            label_flag[0] = True
        else:
            plt.plot(X[i,0], X[i,1], point_type)
    else:
        point_type = 'ro' if (slackvar[i].value) > 1e-07 else 'bo'

        if label_flag[1] == False and point_type == 'bo':        
            plt.plot(X[i,0], X[i,1], point_type, label='-1')
            label_flag[1] = True
        else:
            plt.plot(X[i,0], X[i,1], point_type)

x = np.arange(np.amin(X[:,0]), np.amax(X[:,0]), 0.001)
beta = np.array(beta.value).reshape(-1).tolist()
beta_0 = beta_0.value
y = -beta_0/beta[1] - beta[0]*x/beta[1]
plt.plot(x, y, 'k-')

y_margin1 = (1 - beta_0 - beta[0]*x)/beta[1]
plt.plot(x, y_margin1, 'r--')
y_margin2 = (-1 - beta_0 - beta[0]*x)/beta[1]
plt.plot(x, y_margin2, 'r--')

print("Margin width: {0}".format(2/np.sqrt(beta[0]**2 + beta[1]**2)))
plt.legend(loc='upper right', numpoints=1)
plt.show()
