import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Set parameters for mRNA production and decay
r = 20  # mRNA production rate (min^-1)
gamma = 1 / 1.5  # mRNA decay rate (min^-1)

# Initialize the number of mRNA molecules and set initial condition
m = [0]  # Initial number of mRNA molecules
MaxT = 500  # Number of iterations (time steps)

# Run the Gillespie simulation
tau = []
for i in range(MaxT):
    # Calculate reaction rates
    k1 = r
    k2 = m[i] * gamma
    k0 = k1 + k2
    
    # Time to next reaction
    tau.append(1 / k0 * np.log(1 / np.random.rand()))
    
    # Coin flip to decide the reaction (production or decay)
    CoinFlip = np.random.rand()
    
    if CoinFlip <= k1 / k0:
        m.append(m[i] + 1)  # mRNA production
    else:
        m.append(m[i] - 1)  # mRNA decay

# Calculate time points
T = [0]
for i in range(len(tau)):
    T.append(np.sum(tau[:i + 1]))

# Plot the simulation results with deterministic solution
plt.figure(1)
plt.plot(T, m, label="Stochastic Simulation")
plt.plot(T, r / gamma * (1 - np.exp(-np.array(T) * gamma)), '-k', label="Deterministic Solution")
plt.xlabel('Time (min)')
plt.ylabel('Number of mRNA molecules')
plt.legend()
plt.show()

# Clear variables for steady-state distribution calculation
m = [round(r / gamma)]  # Initial condition at steady state mean
MaxT = 10000  # More iterations for steady state

# Run Gillespie simulation for steady-state distribution
tau = []
for i in range(MaxT):
    k1 = r
    k2 = m[i] * gamma
    k0 = k1 + k2
    
    # Time for the next reaction (average value)
    tau.append(1 / k0)
    
    CoinFlip = np.random.rand()
    
    if CoinFlip <= k1 / k0:
        m.append(m[i] + 1)  # mRNA production
    else:
        m.append(m[i] - 1)  # mRNA decay

# Calculate mean, second moment, and variance
MeanM = np.sum(np.array(m[:-1]) * tau) / np.sum(tau)
SecondMoment = np.sum((np.array(m[:-1]) ** 2) * tau) / np.sum(tau)
VarianceM = SecondMoment - MeanM ** 2

print(f"Mean of mRNA: {MeanM}")
print(f"Variance of mRNA: {VarianceM}")

# Compute probability distribution of mRNA molecules
MaxmRNA = max(m)
# Initialize probability distribution vector
p = np.zeros(MaxmRNA)

# Adjust the size of `p` dynamically when `m[i]` exceeds the current size of `p`
for i in range(MaxT):
    if m[i] >= len(p):
        # Expand the size of p to accommodate the larger mRNA count
        p = np.append(p, np.zeros(m[i] - len(p) + 1))
    
    # Accumulate the time over which a certain number of mRNA molecules existed
    p[m[i]] += tau[i]

# Normalize the distribution
p /= np.sum(p)

# Plot histogram of the simulation's steady-state distribution vs Poisson distribution
plt.figure(2)
plt.bar(range(1, MaxmRNA + 1), p[1:], label="Simulation")
X = np.arange(0, MaxmRNA)
Y = poisson.pmf(X, r / gamma)
plt.plot(X, Y, '-r', label="Poisson Distribution")
plt.xlabel('Number of mRNA molecules')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Initialize the number of mRNA molecules and set initial condition
m = [0]  # Initial number of mRNA molecules
MaxT = 500  # Number of iterations (time steps)

#part (b) This will show tht bias of production/decay events 
tau = []
Bias = []

# Run the Gillespie simulation
for i in range(MaxT):
    # Calculate reaction rates
    k1 = r
    k2 = m[i] * gamma
    k0 = k1 + k2
    
    # Time to next reaction
    tau.append(1 / k0 * np.log(1 / np.random.rand()))
    
    # Calculate the bias of production versus decay
    Bias.append(k1 / k0)
    
    # Coin flip to decide the reaction (production or decay)
    CoinFlip = np.random.rand()
    
    if CoinFlip <= k1 / k0:
        m.append(m[i] + 1)  # mRNA production
    else:
        m.append(m[i] - 1)  # mRNA decay

# Calculate time points
T = np.cumsum(tau)

# Plot the bias as a function of time
plt.figure()
plt.plot(T, Bias, 'b')
plt.xlabel('Time (min)')
plt.ylabel('Bias (Production/Total Events)')
plt.title('Bias of Production vs Decay Events as a Function of Time')
plt.show()

#part (c) this is plotting the time step or tau with respect to time

# Plot the timestep as a function of time
plt.figure()
plt.plot(T, tau, 'r')
plt.xlabel('Time (min)')
plt.ylabel('Timestep (min)')
plt.title('Timestep as a Function of Time')
plt.show()

#This is part (d)

# Set initial conditions
NA = 100  # Initial number of A molecules
NB = 100  # Initial number of B molecules
NC = 0    # Initial number of C molecules
MaxT = 1000  # Total number of reactions
k = 0.1   # Reaction rate constant

# Initialize arrays to store molecule counts
A = [NA]
B = [NB]
C = [NC]
tau = []

# Run the Gillespie simulation for A + B -> C reaction
for i in range(MaxT):
    # Reaction rate for A + B -> C
    k0 = k * A[i] * B[i]
    
    # Time to the next reaction
    tau.append(1 / k0 * np.log(1 / np.random.rand()))
    
    # Update molecule counts
    A.append(A[i] - 1)
    B.append(B[i] - 1)
    C.append(C[i] + 1)
    
    # Stop if A or B are depleted
    if A[-1] <= 0 or B[-1] <= 0:
        break

# Calculate time points
T = np.cumsum(tau)

# Plot the results
plt.figure()
plt.plot(T, A[:len(T)], 'b', label="A")
plt.plot(T, B[:len(T)], 'g', label="B")
plt.plot(T, C[:len(T)], 'r', label="C")
plt.xlabel('Time (min)')
plt.ylabel('Number of molecules')
plt.legend()
plt.title('A + B -> C Reaction Simulation')
plt.show()
