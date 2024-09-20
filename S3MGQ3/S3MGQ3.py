import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the toggle function
def toggle(y, t):
    alpha1 = 10.0
    alpha2 = 10.0
    beta = 2.0
    gamma = 2.0
    
    u, v = y
    du = -u + alpha1 / (1 + v**beta)
    dv = -v + alpha2 / (1 + u**gamma)
    
    return [du, dv]

# Time span
t = np.linspace(0, 15, 500)  # Time from 0 to 15, 500 points

# Initial conditions (0.1, 1) and (5, 4)
initial_conditions_1 = [0.1, 1]
initial_conditions_2 = [5, 4]

# Solving ODEs for both initial conditions
solution_1 = odeint(toggle, initial_conditions_1, t)
solution_2 = odeint(toggle, initial_conditions_2, t)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot for initial conditions (0.1, 1)
plt.subplot(2, 1, 1)
plt.plot(t, solution_1[:, 0], label='u(t)')
plt.plot(t, solution_1[:, 1], label='v(t)')
plt.title('Solution for initial conditions (u0, v0) = (0.1, 1)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()

# Plot for initial conditions (5, 4)
plt.subplot(2, 1, 2)
plt.plot(t, solution_2[:, 0], label='u(t)')
plt.plot(t, solution_2[:, 1], label='v(t)')
plt.title('Solution for initial conditions (u0, v0) = (5, 4)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
