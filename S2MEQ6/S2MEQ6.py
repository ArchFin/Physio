import numpy as np
import matplotlib.pyplot as plt

# Parameters for the system didn't care enough for them to be specific 
kD = 1.0  # Delta activation rate
kN = 1.0  # Notch inhibition rate
gamma_N = 1.0  # Notch decay rate
gamma_D = 0.1  # Delta decay rate

# Activation functions
def F(D):
    """ Notch activation as a function of Delta. """
    return kD * D**2 / (0.1 + D**2)

def G(N):
    """ Delta inhibition as a function of Notch. """
    return kN / (1 + 10 * N**2)

# System of equations for dD1/dt and dD2/dt
def delta_dynamics(D1, D2):
    """ Dynamical equations for Delta in the two cells. """
    N1 = F(D2) / gamma_N
    N2 = F(D1) / gamma_N
    dD1_dt = G(N1) - gamma_D * D1
    dD2_dt = G(N2) - gamma_D * D2
    return dD1_dt, dD2_dt

# Create a grid for D1 and D2 values
D1_vals = np.linspace(0, 2, 20)
D2_vals = np.linspace(0, 2, 20)
D1, D2 = np.meshgrid(D1_vals, D2_vals)

# Compute the derivatives at each point on the grid
dD1_dt, dD2_dt = delta_dynamics(D1, D2)

