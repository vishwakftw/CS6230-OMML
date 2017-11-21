import json
import torch as t
import numpy as np
import torch.nn as nn
from torchvision import transforms
from argparse import ArgumentParser
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.autograd import Variable as V
from utils import return_model, return_optimizer

np.random.seed(29)
t.manual_seed(29)

# Get the parameters
p = ArgumentParser()
p.add_argument('--opt', required=True, type=str, help='Optimizer --> adam | adagrad | rmsprop | nag | cm')
p.add_argument('--opt_params', required=True, type=str, help='File containing params in json format')
p.add_argument('--maxiter', default=int(5e04), type=int, help='Maximum iterations')
p.add_argument('--architecture', required=True, type=str, help='CSV file with number of nodes per layer')
p.add_argument('--activation', default='relu', type=str, help='Activation function to be used')
p.add_argument('--dataset', required=True, type=str, help='Dataset to use --> mnist | faces')
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

elif p.dataset == 'faces':
    # Do something
