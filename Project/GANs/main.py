import sys
import DCGAN as dc

# Dataset
dset = sys.argv[1]
root = sys.argv[2]

# Get dataset
if dset == 'MNIST':
    transformations = [transforms.ToTensor()]
    transformations.append(transforms.Normalize((0.5, ), (0.5, )))
    dataset = dset.MNIST(root=root, download=True, transform=transforms.Compose(transformations))
    n_chan = 1
elif dset == 'CIFAR10':
    transformations = [transforms.ToTensor()]
    transformations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    dataset = dset.CIFAR10(root=root, download=True, transform=transforms.Compose(transformations))
    n_chan = 3
    
# DCGAN object initialization
n_z = 128
ngpu = 1
loss = 'BCE'

Gen_model = dc.DCGAN(arch=dset, n_z=n_z, ngpu=ngpu, loss=loss)

# DCGAN training scheme
batch_size = 100
n_iters = int(5e04)
opt_dets = {'gen':    {'name' : 'adam',
                       'learn_rate' : 1e-04,
                       'betas' : (0.5, 0.99)
                      },
            'dis':    {'name' : 'adam',
                       'learn_rate' : 1e-04,
                       'momentum' : (0.5, 0.99)
                      }
           }

# Optional arguments
show_period = 500
display_images = True
misc_options = ['init_scheme', 'save_model']

# Call training
Gen_model.train(dataset=dataset, batch_size=batch_size, n_iters=n_iters, optimizer_details=opt_dets, show_period=show_period, display_images=display_images, misc_options=misc_options)
