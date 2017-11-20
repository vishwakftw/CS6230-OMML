import torch as t
import torch.nn as nn
import torch.optim as optim

def return_model(arch_vals, activation):
    activations = {'relu': nn.ReLU(True), 
                   'sigmoid': nn.Sigmoid(),
                   'tanh': nn.Tanh(),
                   'leaky_relu': nn.LeakyReLU(0.2, inplace=True)
                  }
    model = nn.Sequential()
    for i in range(0, len(arch_vals) - 1):
        model.add_module('linear_{0}-{1}-{2}'.format(i, arch_vals[i], arch_vals[i+1]), nn.Linear(arch_vals[i], arch_vals[i+1]))
        if i != len(arch_vals) - 2:
            model.add_module('{0}_{1}'.format(activation, i), activations[activation])
    return model

def return_optimizer(optimizer, opt_params, model_params):
    if optimizer == 'adam':
        opt_function = optim.Adam(params=model_params, lr=opt_params['lr'])
    if optimizer == 'adagrad':
        opt_function = optim.Adagrad(params=model_params, lr=opt_params['lr'])
    if optimizer == 'rmsprop':
        opt_function = optim.RMSprop(params=model_params, lr=opt_params['lr'])
    if optimizer == 'nag':
        opt_function = optim.SGD(params=model_params, lr=opt_params['lr'], momentum=opt_params['momentum'], nesterov=True)
    if optimizer == 'cm':
        opt_function = optim.SGD(params=model_params, lr=opt_params['lr'], momentum=opt_params['momentum'])

    if 'betas' in opt_function.param_groups[0].keys():
        opt_function.param_groups[0]['betas'] = opt_params['betas']
    if 'alpha' in opt_function.param_groups[0].keys():
        opt_function.param_groups[0]['alpha'] = opt_params['alpha']
    if 'weight_decay' in opt_params.keys():
        opt_function.param_groups[0]['weight_decay'] = opt_params['weight_decay']
    return opt_function
