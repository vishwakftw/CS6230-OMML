import json
import torch as t
import numpy as np
import torch.nn as nn
from torchvision import transforms
from argparse import ArgumentParser
from torchvision import utils as tv_u
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from utils import return_optimizer

np.random.seed(29)
t.manual_seed(29)

# Get the parameters
p = ArgumentParser()
p.add_argument('--opt', required=True, type=str, help='Optimizer --> adam | adagrad | rmsprop | nag | cm')
p.add_argument('--opt_params', required=True, type=str, help='File containing params in json format')
p.add_argument('--maxiter', default=int(5e04), type=int, help='Maximum iterations')
p.add_argument('--architecture', required=True, type=str, help='CSV file with number of nodes per layer')
p.add_argument('--init', default='random', type=str, help='Initialization scheme to use --> random | he | xavier')
p.add_argument('--dataset', required=True, type=str, help='Dataset to use --> mnist')
p.add_argument('--dataroot', default='./', type=str, help='Data root folder')
p.add_argument('--cuda', default=-1, type=int, help='Enter CUDA device (> 0), or -1 if no CUDA')
p = p.parse_args()

# CUDA check
if p.cuda != -1:
    t.cuda.set_device(p.cuda)

# Load the data
if p.dataset == 'mnist':
    transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Lambda(lambda x: x.view(784))
                                        ])
    tr_dset = MNIST(root=p.dataroot, train=True, transform=transformations, download=True)
    te_dset = MNIST(root=p.dataroot, train=False, transform=transformations, download=True)

else:
    raise ValueError("No other dataset available")

tr_d_loader = DataLoader(dataset=tr_dset, batch_size=64, shuffle=True)
te_d_loader = DataLoader(dataset=te_dset, batch_size=5000, shuffle=True)

# Build MLP architecture
arch_vals = np.genfromtxt(p.architecture, delimiter=',').reshape(-1).astype(int).tolist()
model = nn.Sequential()
for i in range(0, len(arch_vals) - 1):
    model.add_module('linear_{0}-{1}-{2}'.format(i, arch_vals[i], arch_vals[i+1]), nn.Linear(arch_vals[i], arch_vals[i+1]))
    if len(arch_vals) % 2 == 1:
        if i != (len(arch_vals) - 1)/2 and i != len(arch_vals) - 2:
            model.add_module('sigmoid_{0}'.format(i), nn.Sigmoid())
    else:
        if i != (len(arch_vals)/2 - 1):
            model.add_module('sigmoid_{0}'.format(i), nn.Sigmoid())
model.add_module('sigmoid_{0}'.format(i+1), nn.Sigmoid())

if p.init == 'xavier':
    for param in model.parameters():
        if len(param.size()) >= 2:
            param.data = nn.init.xavier_uniform(param.data)
elif p.init == 'he':
    for param in model.parameters():
        if len(param.size()) >= 2:
            param.data = nn.init.kaiming_uniform(param.data)
print(model)

if p.cuda != -1:
    model = model.cuda()

# Build Optimizer
optimizer = return_optimizer(p.opt, json.load(open(p.opt_params)), model.parameters())
print(optimizer)

# Loss function
loss_fn = nn.BCELoss()
if p.cuda != -1:
    loss_fn = loss_fn.cuda()

optimizer_params = json.load(open(p.opt_params))
params = [p.dataset]
for k in sorted(list(optimizer_params)):
    params.append(optimizer_params[k])
for k in arch_vals:
    params.append(k)
params += ['sigmoid', p.init]

loss_log = open('./{0}/loss_autoencoder_{1}.txt'.format(p.opt, params), 'w')

flag = False
iters = 0
while flag != True:
    model.train()
    for i, itr in enumerate(tr_d_loader):
        x = itr[0]
        if p.cuda != -1:    
            x = x.cuda()
        x = V(x)
        cur_loss = loss_fn(model(x), x)
        loss_log.write('{0}\t{1}\n'.format(iters, round(cur_loss.data[0], 6)))
        if iters == p.maxiter:
            flag = True
            break
        if iters % 1000 == 0:
            print('{0} iterations completed'.format(iters))
        cur_loss = cur_loss.cuda()
        model.zero_grad()
        cur_loss.backward()
        optimizer.step()
        iters += 1

model.eval()
N = 0
te_loss = 0.0
for i, itr in enumerate(te_d_loader):
    x = itr[0]
    if p.cuda != -1:    
        x = x.cuda()
    x = V(x)
    te_loss += loss_fn(model(x), x).data[0]*(itr[0].size(0))
    N += itr[0].size(0)
te_loss = te_loss/N
print('Test Error after {0} iterations with {1} is {2}'.format(p.maxiter, p.opt, round(te_loss, 5)))
