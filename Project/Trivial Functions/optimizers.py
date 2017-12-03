import numpy as np

def GD(t, X, G, lr):
    """
        Function to perform one GD step
        Args:
            t   => previous timestep
            X   => Current point
            G   => Gradient at current point
            lr  => Learning Rate 
    """
    timestep = t + 1
    update = X - lr*G
    return timestep, update

def CM(t, X, G, V, m, lr):
    """
        Function to perform one Classical Momentum step
        Args:
            t   => previous timestep
            X   => Current point
            G   => Gradient at current point
            V   => Velocity at the previous iteration
            m   => Momentum Parameter
            lr  => Learning Rate
    """
    timestep = t + 1
    velocity = m*V - lr*G
    update = X + velocity
    return timestep, update, velocity

def Adam(t, X, G, FM, SM, betas=(0.9, 0.99), lr, eps=1e-08):
    """
        Function to perform one Adam step
        Args:
            t       => previous timestep
            X       => Current point
            G       => Gradient at current point
            FM      => First Moment at the previous iteration
            SM      => Second Moment at the previous iteration
            betas   => Beta parameter defined in Adam
            lr      => Learning Rate
            eps     => (optional, default=1e-08) Prevent divide by 0 error
    """
    timestep = t + 1
    FM = betas[0]*FM + (1 - betas[0])*G
    SM = betas[1]*SM + (1 - betas[1])*G*G
    mh = FM/(1 - betas[0]**timestep)
    vh = SM/(1 - betas[1]**timestep)
    update = X - lr*mh/(np.sqrt(vh) + eps)
    return timestep, update, FM, SM

def Adagrad(t, X, G, SSG, lr, eps=1e-08):
    """
        Function to perform one Adagrad step
        Args:
            t   => previous timestep
            X   => Current point
            G   => Gradient at current point
            SSG => Sum of squared gradient till previous timestep
            lr  => Learning Rate
            eps => (optional, default=1e-08) Prevent divide by 0 error
    """
    timestep = t + 1
    update = X - lr*G/(np.sqrt(SSG) + eps)
    SSG = SSG + G*G
    return timestep, update, SSG

def Adadelta(t, X, G, EG, EDX, lr, rho=0.9, eps=1e-08):
    """
        Function to perform one Adadelta step
        Args:
            t   => previous timestep
            X   => Current point
            G   => Gradient at current point
            EG  => Running average of squared gradients till previous timestep
            EDX => Running average of squared updates till previous timestep
            lr  => Learning Rate
            eps => (optional, default=1e-08) Prevent divide by 0 error
    """
    timestep = t + 1
    EG = rho*EG + (1 - rho)*G*G
    delx = -np.sqrt(EDX + eps)*G/np.sqrt(EG + eps)
    EDX = rho*EDX + (1 - rho)*delx*delx
    update = X + delx
    return timestep, update, EG, EDX

def RMSprop(t, X, G, EG, lr, eps=1e-08):
    """
        Function to perform one Adadelta step
        Args:
            t   => previous timestep
            X   => Current point
            G   => Gradient at current point
            EG  => Running average of squared gradients till previous timestep
            lr  => Learning Rate
            eps => (optional, default=1e-08) Prevent divide by 0 error
    """
    timestep = t + 1
    EG = 0.9*EG + 0.1*G*G
    update = X - lr*G/np.sqrt(EG + eps)
    return timestep, update, EG

_opt_dicts = {'GD': GD, 'CM': CM, 'Adam': Adam, 'Adagrad': Adagrad, 'Adadelta': Adadelta, 'RMSprop': RMSprop}
