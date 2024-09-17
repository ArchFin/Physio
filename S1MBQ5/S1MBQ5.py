import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_species = 100  # number of species
num_steps = 10000  # number of simulation steps

# Initialize species with random fitness values between 0 and 1
fitness = np.random.rand(num_species)

# Arrays to store the minimum fitness and the maximum of all previous Bmins
bmin_list = []
bmin_max_list = []

# Simulate the Bak-Sneppen dynamics
bmin_max = 0  # Initialize the maximum of all previous Bmins
for _ in range(num_steps):
    # Find the species with the minimum fitness
    bmin_idx = np.argmin(fitness)
    bmin_value = fitness[bmin_idx]
    
    # Update the species with the minimum fitness and one of its neighbors
    fitness[bmin_idx] = np.random.rand()  # Reset the minimum fitness
    neighbor_idx = (bmin_idx + np.random.choice([-1, 1])) % num_species  # Select one random neighbor
    fitness[neighbor_idx] = np.random.rand()  # Reset the fitness of the neighbor
    
    # Track the minimum fitness at each step
    bmin_list.append(bmin_value)
    
    # Track the maximum of all previous Bmins
    bmin_max = max(bmin_max, bmin_value)
    bmin_max_list.append(bmin_max)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(bmin_list, label='$B_{min}$ (current minimum)', alpha=0.7)
plt.plot(bmin_max_list, label='$\max(B_{min})$ (maximum over time)', color='red', alpha=0.7)
plt.title("Bak-Sneppen Model: $B_{min}$ vs Time and Maximum Envelope")
plt.xlabel("Time Step")
plt.ylabel("$B_{min}$")
plt.legend()
plt.grid(True)
plt.show()