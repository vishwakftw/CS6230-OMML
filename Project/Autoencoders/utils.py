import torch as t
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, net_params, init_scheme):
        """
            Class to produce models based on JSON objects
        """
        super(Model, self).__init__()

        activations = {'relu': nn.ReLU(True), 
                       'sigmoid': nn.Sigmoid(),
                       'tanh': nn.Tanh(),
                       'leaky_relu': nn.LeakyReLU(0.2, inplace=True)
                      }
        self.conv_part = None
        self.mlp_part = None
        for i in range(0, len(net_params)):
            if net_params[i]['type'] == 'CONV':
                if self.conv_part is None:
                    self.conv_part = nn.Sequential()
                in_ = net_params[i]['params']['in']
                out_ = net_params[i]['params']['out']
                k_size = net_params[i]['params']['kernel']
                stride = net_params[i]['params']['stride']
                pad = net_params[i]['params']['padding']
                self.conv_part.add_module('conv_{0}-{1}-{2}'.format(i, in_, out_), 
                                          nn.Conv2d(in_channels=in_, out_channels=out_,
                                                    kernel_size=k_size, stride=stride,
                                                    padding=pad))
                if 'activation' in net_params[i].keys():    
                    self.conv_part.add_module('{0}_{1}'.format(i, net_params[i]['activation']), activations[net_params[i]['activation']])
            elif net_params[i]['type'] == 'MLP':
                if self.mlp_part is None:
                    self.mlp_part = nn.Sequential()
                in_ = net_params[i]['params']['in']
                out_ = net_params[i]['params']['out']
                self.mlp_part.add_module('linear_{0}-{1}-{2}'.format(i, in_, out_),
                                         nn.Linear(in_, out_))
                if 'activation' in net_params[i].keys():    
                    self.mlp_part.add_module('{0}_{1}'.format(i, net_params[i]['activation']), activations[net_params[i]['activation']])

        if init_scheme == 'xavier':
            for p in self.mlp_part.parameters():
                if len(p.size()) >= 2:
                    p.data = nn.init.xavier_uniform(p.data)
        elif init_scheme == 'he':
            for p in self.mlp_part.parameters():
                if len(p.size()) >= 2:
                    p.data = nn.init.kaiming_uniform(p.data)

    def forward(self, input):
        if self.conv_part is not None:
            pass_ = self.conv_part(input)
        pass_ = pass_.view(self.batch_size, -1)
        if self.mlp_part is not None:
            pass_ = self.mlp_part(pass_)
        return pass_

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
def return_optimizer(optimizer, opt_params, model_params):
    if optimizer == 'adam':
        opt_function = optim.Adam(params=model_params, lr=opt_params['lr'])
    if optimizer == 'adagrad':
        opt_function = optim.Adagrad(params=model_params, lr=opt_params['lr'])
    if optimizer == 'adadelta':
        opt_function = optim.Adadelta(params=model_params, lr=opt_params['lr'])
    if optimizer == 'rmsprop':
        opt_function = optim.RMSprop(params=model_params, lr=opt_params['lr'])
    if optimizer == 'nag':
        opt_function = optim.SGD(params=model_params, lr=opt_params['lr'], momentum=opt_params['momentum'], nesterov=True)
    if optimizer == 'cm':
        opt_function = optim.SGD(params=model_params, lr=opt_params['lr'], momentum=opt_params['momentum'])

    if 'beta1' in opt_function.param_groups[0].keys():
        opt_function.param_groups[0]['betas'][0] = opt_params['beta1']
    if 'beta2' in opt_function.param_groups[0].keys():
        opt_function.param_groups[0]['betas'][1] = opt_params['beta2']
    if 'weight_decay' in opt_params.keys():
        opt_function.param_groups[0]['weight_decay'] = opt_params['weight_decay']
    return opt_function
