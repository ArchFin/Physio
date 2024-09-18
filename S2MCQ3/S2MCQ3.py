import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
V0 = -50        # Height of the initial voltage pulse.
Sigma = 1       # Width of the initial voltage pulse.
tau = 2
lambda_ = 9.1 / 50
VNa = 54        # Na Nernst voltage.
VK = -75        # K Nernst voltage.
gNa_base = 1 / 5  # Na conductivity when the channels are inactivated.
gK = 5          # K conductivity.
betaq = 0.5     # beta*q from eqn. 17.29.
Vthresh = -40   # Threshold value V* from eqn. 17.29.
L = 10          # Length over which we will integrate (mm).
dx = 0.01       # Size of our circuit elements.
t = 30          # Total integration time (ms).
dt = 0.0005     # Time step used for the integration.
SavePeriod = 100 # How often we want to save the results.

# Derived parameters
xRange = np.arange(-L + dx, L, dx)
tRange = np.arange(0, t + dt, dt)
CourantNumber = lambda_ / tau * dt / dx
Vmem = (VNa * gNa_base + VK * gK) / (gNa_base + gK)  # Steady state membrane potential.

# Initial voltage pulse
V = np.zeros((len(xRange), 2))  # Array to store voltage at each time and space step.
V[:, 0] = (V0 - Vmem) * np.exp(-((xRange) / Sigma) ** 2) + Vmem  # Initial condition
Vout = V[:, 0:1]  # Store the results
tout = [tRange[0]]  # Store time points

SaveCounter = 0

# Run the integration
for i in tqdm(range(1, len(tRange))):  # Progress bar for the time loop
    # Sodium conductivity function for the current voltage
    gNaFunc = 100 / (1 + np.exp(betaq * (Vthresh - V[:, 0]))) + gNa_base

    # Update voltage at the first circuit element (j = 1)
    V[0, 1] = V[0, 0] + dt / tau * (lambda_ ** 2 / dx ** 2 * (V[1, 0] - 2 * V[0, 0] + V[-1, 0])
                                     - gNaFunc[0] / gK * (V[0, 0] - VNa) - (V[0, 0] - VK))

    # Update voltage for the middle circuit elements (j = 2 to j = len(V)-1)
    for j in range(1, len(xRange) - 1):
        V[j, 1] = V[j, 0] + dt / tau * (lambda_ ** 2 / dx ** 2 * (V[j + 1, 0] - 2 * V[j, 0] + V[j - 1, 0])
                                         - gNaFunc[j] / gK * (V[j, 0] - VNa) - (V[j, 0] - VK))

    # Update voltage at the last circuit element
    V[-1, 1] = V[-1, 0] + dt / tau * (lambda_ ** 2 / dx ** 2 * (V[0, 0] - 2 * V[-1, 0] + V[-2, 0])
                                       - gNaFunc[-1] / gK * (V[-1, 0] - VNa) - (V[-1, 0] - VK))

    # Save the results periodically
    if SaveCounter == SavePeriod:
        Vout = np.column_stack((Vout, V[:, 1]))
        tout.append(tRange[i])
        SaveCounter = 0

    SaveCounter += 1
    V[:, 0] = V[:, 1]  # Shift the voltage in time
    V[:, 1] = 0  # Reset the current time step

# Plot the profiles for different time points based on actual saved data
time_indices = [0, 100, 200, 300, 400, 500]  # Choose indices that are valid

# Ensure the indices don't exceed the size of Vout
time_indices = [i for i in time_indices if i < Vout.shape[1]]

for idx in time_indices:
    plt.plot(xRange, Vout[:, idx], label=f't={idx}')

plt.xlim([0, L])
plt.legend()
plt.xlabel('x (mm)')
plt.ylabel('Voltage (mV)')
plt.title('Voltage Profiles at Different Time Points')
plt.show()
