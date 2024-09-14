import numpy as np
from matplotlib import pyplot as plt
import sympy.solvers.ode as ode
from sympy import Function, Derivative, Symbol
from scipy.stats import linregress


#Solution to part (iii)
def sn(N, b, c, sb):
    return b**(N)*sb + ((b**(N)-1)/(b-1))*c

#getting number values
N = np.geomspace(1e0,1e2,1) # geometric spacing
b = 2 
c = 0.3
sb = 1

s_n = sn(N, b, c, sb)

#setting up the figure for visualisation
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(180/25.4,50/25.4)
ax = axs[0]
ax.set_xscale('log')
ax.set_xlabel('$N$')
ax.set_ylabel('dN/dt')
ax.plot(N,s_n)
ax = axs[1]
ax.set_xscale('log')
ax.set_xlabel('$N$')
ax.set_yscale('log')
ax.set_ylabel('|dN/dt|')
ax.plot(N,np.abs(s_n))
fig.tight_layout()
plt.show()