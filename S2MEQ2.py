import numpy as np
import matplotlib.pyplot as plt

# Parameters for the logistic growth model
f0 = 1.0  # intrinsic growth rate
p = 100  # carrying capacity
# Population size range and logistic growth model
N = np.linspace(-20, 120, 400)  # extended range for better visualization
dN_dt = f0 * N * (1 - N / p)

# Meshgrid for quiver plot (arrows indicating direction of flow)
N_mesh = np.linspace(-10, 110, 10)
dN_mesh = np.linspace(0, 0, 10)
X, Y = np.meshgrid(N_mesh, dN_mesh)

# Velocity components for arrows
U = np.sign(X * (1 - X / p))  # direction based on the logistic equation
V = 0 * X  # no vertical flow in this 1D phase plot

# Create the plot
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(N, dN_dt, label=r'$\frac{dN}{dt} = f0N(1 - \frac{N}{p})$')

# Fixed points (extinction and carrying capacity)
ax.plot(0, 0, marker='o', color='red')  # Extinction point
ax.plot(p, 0, marker='o', color='green')  # Carrying capacity

# Quiver plot for arrows indicating direction
ax.quiver(X, Y, U, V, units='width')

# Labels and title
ax.set_xlabel('Population size (N)')
ax.set_ylabel('Rate of Change (dN/dt)')
ax.axhline(0, color='black',linewidth=0.8)  # Horizontal axis
ax.axvline(p, color='green', linestyle='--', label="Carrying capacity (p)")  # Carrying capacity
ax.axvline(0, color='red', linestyle='--', label="Extinction (N=0)")  # Extinction point
ax.legend()

# Show the plot
plt.grid(True)
plt.title('Logistic Growth Phase Space with Flow Arrows')
plt.show()