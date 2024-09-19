import numpy as np
import matplotlib.pyplot as plt

# Parameters for the logistic growth model
f0 = 1.0  # intrinsic growth rate
p = 100  # carrying capacity
N = np.linspace(0, 120, 400)  # population size values

# Logistic growth model: dN/dt = f0(1 - N/p)N
dN_dt = f0 * N * (1 - N / p)

# Plotting the phase space
plt.figure(figsize=(8, 6))
plt.plot(N, dN_dt, label=r'$\frac{dN}{dt} = f0(1 - \frac{N}{p})N$')
plt.axhline(0, color='black',linewidth=0.8)  # Horizontal axis
plt.axvline(p, color='green', linestyle='--', label="Carrying capacity (p)")  # Carrying capacity
plt.axvline(0, color='red', linestyle='--', label="Extinction (N=0)")  # Extinction point

# Labels and title
plt.title('Phase Space of Logistic Growth Model')
plt.xlabel('Population size (N)')
plt.ylabel('Rate of Change (dN/dt)')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()