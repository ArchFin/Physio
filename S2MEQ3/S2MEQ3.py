import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define parameters
p_out = [34.9133, 3.8566, 0.4807, 0.0248, 0.9272, 0.0276]
alpha = p_out[2]
beta = p_out[3]
gamma = p_out[4]
delta = p_out[5]

# Define the predator-prey system
def lotka_volterra(y, t, alpha, beta, gamma, delta):
    x, z = y
    dxdt = alpha * x - beta * x * z
    dzdt = delta * x * z - gamma * z
    return [dxdt, dzdt]

# Generate phase portrait (vector field)
x1, x2 = np.meshgrid(np.arange(0, 80, 4), np.arange(0, 80, 4))
x1dot = alpha * x1 - beta * x1 * x2
x2dot = delta * x1 * x2 - gamma * x2

plt.figure(figsize=(8, 6))
plt.quiver(x1, x2, x1dot, x2dot)
plt.xlabel('Prey Population (Hares)')
plt.ylabel('Predator Population (Lynxes)')
plt.title('Phase Portrait with Nullclines')
plt.grid(True)

# Plot nullclines
# Prey nullcline: alpha * x - beta * x * y = 0 -> y = alpha / beta
prey_nullcline = alpha / beta
plt.axhline(prey_nullcline, color='red', linestyle='--', label='Prey Nullcline')

# Predator nullcline: delta * x * y - gamma * y = 0 -> x = gamma / delta
predator_nullcline = gamma / delta
plt.axvline(predator_nullcline, color='green', linestyle='--', label='Predator Nullcline')

plt.legend()

# Solve the system of ODEs
t = np.linspace(0, 20, 200)  # time points
y0 = [p_out[0], p_out[1]]    # initial conditions (prey, predator)

solution = odeint(lotka_volterra, y0, t, args=(alpha, beta, gamma, delta))

# Plot prey and predator population over time
plt.figure(figsize=(8, 6))
plt.plot(t, solution[:, 0], label='Prey (Hares)', color='g')
plt.plot(t, solution[:, 1], label='Predator (Lynxes)', color='k')
plt.xlabel('Time (years)')
plt.ylabel('Population')
plt.title('Time Dependence of Predator-Prey Populations')
plt.legend()
plt.grid(True)

# Plot the predator-prey trajectory in phase space
plt.figure(figsize=(8, 6))
plt.plot(solution[:, 0], solution[:, 1], '--r')
plt.xlabel('Prey Population (Hares)')
plt.ylabel('Predator Population (Lynxes)')
plt.title('Predator vs Prey Phase Space')
plt.grid(True)

plt.show()
