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

def Adam(t, X, G, FM, SM, betas, lr, eps=1e-08):
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
