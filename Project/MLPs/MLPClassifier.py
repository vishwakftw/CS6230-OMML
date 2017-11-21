import json
import torch as t
import numpy as np
import torch.nn as nn
from torchvision import transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from utils import return_model, return_optimizer
from torchvision.datasets import MNIST, CIFAR10, SVHN

np.random.seed(29)
t.manual_seed(29)

# Get the parameters
p = ArgumentParser()
p.add_argument('--opt', required=True, type=str, help='Optimizer --> adam | adagrad | rmsprop | nag | cm')
p.add_argument('--opt_params', required=True, type=str, help='File containing params in json format')
p.add_argument('--maxiter', default=int(5e04), type=int, help='Maximum iterations')
p.add_argument('--architecture', required=True, type=str, help='CSV file with number of nodes per layer')
p.add_argument('--activation', default='relu', type=str, help='Activation function to be used')
p.add_argument('--init', default='random', type=str, help='Initialization scheme to use --> random | he | xavier')
p.add_argument('--dataset', required=True, type=str, help='Dataset to use --> mnist | cifar10 | svhn')
p.add_argument('--dataroot', default='./', type=str, help='Data root folder')
p.add_argument('--cuda', default=-1, type=int, help='Enter CUDA device (> 0), or -1 if no CUDA')
p = p.parse_args()

# CUDA check
if p.cuda != -1:
    t.cuda.set_device(p.cuda)

# Load the data
if p.dataset == 'mnist':
    transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize((0.5,), (0.5,)), 
                                          transforms.Lambda(lambda x: x.view(784))
                                        ])
    tr_dset = MNIST(root=p.dataroot, train=True, transform=transformations, download=True)
    te_dset = MNIST(root=p.dataroot, train=False, transform=transformations, download=True)

elif p.dataset == 'cifar10':
    transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                          transforms.Lambda(lambda x: x.view(3072))
                                        ])
    tr_dset = CIFAR10(root=p.dataroot, train=True, transform=transformations, download=True)
    te_dset = CIFAR10(root=p.dataroot, train=False, transform=transformations, download=True)

elif p.dataset == 'svhn':
    transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                          transforms.Lambda(lambda x: x.view(3072))
                                        ])
    tr_dset = SVHN(root=p.dataroot, split='train', transform=transformations, download=True)
    te_dset = SVHN(root=p.dataroot, split='test', transform=transformations, download=True)
                                    
tr_d_loader = DataLoader(dataset=tr_dset, batch_size=64, shuffle=True)
te_d_loader = DataLoader(dataset=te_dset, batch_size=5000, shuffle=True)

# Build MLP architecture
arch_vals = np.genfromtxt(p.architecture, delimiter=',').reshape(-1).astype(int).tolist()
model = return_model(arch_vals, p.activation, init=p.init)
print(model)

if p.cuda != -1:
    model = model.cuda()

# Build Optimizer
optimizer = return_optimizer(p.opt, json.load(open(p.opt_params)), model.parameters())
print(optimizer)

# Loss function
loss_fn = nn.CrossEntropyLoss()
if p.cuda != -1:
    loss_fn = loss_fn.cuda()

optimizer_params = json.load(open(p.opt_params))
params = [p.dataset, p.opt]
for k in sorted(list(optimizer_params)):
    params.append(optimizer_params[k])
for k in arch_vals:
    params.append(k)

loss_log = open('loss_classifier_{0}.txt'.format(params), 'w')

flag = False
iters = 0
while flag != True:
    model.train()
    for i, itr in enumerate(tr_d_loader):
        x, y = itr[0], itr[1]
        if p.cuda != -1:    
            x = x.cuda()
            y = y.cuda()
        x, y = V(x), V(y)
        cur_loss = loss_fn(model(x), y)
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
tr_d_loader = DataLoader(dataset=tr_dset, batch_size=5000, shuffle=True)
N = 0
accuracy = 0
for i, itr in enumerate(tr_d_loader):
    x, y = itr[0], itr[1]
    if p.cuda != -1:    
        x = x.cuda()
        y = y.cuda()
    x, y = V(x), V(y)
    forward_pass = model(x)
    maxes, pred_y = forward_pass.max(1)
    pred_y = pred_y.view(-1)
    
    N += itr[1].size(0)
    accuracy += np.linalg.norm(np.array((pred_y - y).data.tolist()), ord=0)
accuracy = (1 - accuracy/N)*100
print('Train Accuracy after {0} iterations with {1} is {2}'.format(p.maxiter, p.opt, round(accuracy, 5)))

N = 0
accuracy = 0
for i, itr in enumerate(te_d_loader):
    x, y = itr[0], itr[1]
    if p.cuda != -1:    
        x = x.cuda()
        y = y.cuda()
    x, y = V(x), V(y)
    forward_pass = model(x)
    maxes, pred_y = forward_pass.max(1)
    pred_y = pred_y.view(-1)
    
    N += itr[1].size(0)
    accuracy += np.linalg.norm(np.array((pred_y - y).data.tolist()), ord=0)
accuracy = (1 - accuracy/N)*100
print('Test Accuracy after {0} iterations with {1} is {2}'.format(p.maxiter, p.opt, round(accuracy, 5)))
