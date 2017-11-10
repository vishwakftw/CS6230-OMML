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
problem.solve(solver=C.ECOS, abstol=1e-10, reltol=1e-09, feastol=1e-10, max_iters=1000)

print("Problem exited with status: {0} and value attained: {1}".format(problem.status, round(problem.value, 5)))

# Saving the \alphas into a text file
alpha_file = open('alpha.txt', 'w')
for i in range(0, m):
    alpha_file.write('{0}\n'.format(round(alpha[i].value, 8)))
alpha_file.close()

# Plotting the comparison of \alpha vs y(<\beta, x> + \beta_0)
alpha_vals = np.around(np.genfromtxt('alpha.txt'), decimals=7)
beta_vals = np.around(np.genfromtxt('beta.txt'), decimals=7)

for i in range(0, m):
    plt.plot(alpha_vals[i], beta_vals[i], 'b.')

plt.xlim((-0.005, 1.005))
plt.xlabel('$\\alpha_{i}$')
plt.ylabel('$y_{i}(\\beta_{i}^{T}x_{i} + \\beta_{0})$')
plt.show()
