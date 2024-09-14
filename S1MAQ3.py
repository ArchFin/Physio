import numpy as np
from matplotlib import pyplot as plt
import sympy.solvers.ode as ode
from sympy import Function, Derivative, Symbol
from scipy.stats import linregress


#Solution to part (iii)
def sn(N, c, sb, a):
    
    for j in a:
        b = (j+1)/2
        s_n = []
        for i in N:
            y = b**(i)*sb + ((b**(i)-1)/(b-1))*c
            s_n.append(y)
        fig, axs = plt.subplots(1)
        fig.set_size_inches(180/25.4,50/25.4)
        axs.set_xlabel('$N$')
        axs.set_ylabel('s_n')
        axs.plot(N,s_n)


#getting number values
N = range(0,10,1) # geometric spacing
a = np.arange(-1,1,0.1)
c = 0.3
sb = 1

s_n = sn(N, c, sb, a)

#setting up the figure for visualisation
# fig, axs = plt.subplots(1)
# fig.set_size_inches(180/25.4,50/25.4)
# axs.set_xlabel('$N$')
# axs.set_ylabel('s_n')
# axs.plot(N,s_n)

plt.show()