#S1MAQ2

import numpy as np
from matplotlib import pyplot as plt
import sympy.solvers.ode as ode
from sympy import Function, Derivative, Symbol
from scipy.stats import linregress


# #Solution to part (i)

# #Defining the function r
# def r(N,r0,K):
#     return r0*(1-(N/K)**2)

# #Solution to part (iii)
# def f(N, f0, P):
#     return f0*(1-(N/P))

# #defining the constants
# r0 = 0.0346
# K = 1000
# f0 = 0.0346
# P = 1000
# #getting number values
# N = np.geomspace(1e0,1e4,1000) # geometric spacing

# dNdt = r(N,r0,K)*N

# dNdt_1 = f(N,f0,P)*N

# #solution to part (iv)
# print(N[np.argmax(dNdt)])
# print(K/np.sqrt(3))
# Nmax = K/np.sqrt(3)
# #setting up the figure for visualisation
# fig, axs = plt.subplots(1, 2)
# fig.set_size_inches(180/25.4,50/25.4)
# ax = axs[0]
# ax.set_xscale('log')
# ax.set_xlabel('$N$')
# ax.set_ylabel('dN/dt')
# ax.plot(N,dNdt)
# # plot analytical solution
# ax.plot(Nmax,r(Nmax,r0,K)*Nmax,marker='o')
# #ax.plot(N,dNdt_1)
# ax = axs[1]
# ax.set_xscale('log')
# ax.set_xlabel('$N$')
# ax.set_yscale('log')
# ax.set_ylabel('|dN/dt|')
# ax.plot(N,np.abs(dNdt))
# # plot analytical solution
# ax.plot(Nmax,np.abs(r(Nmax,r0,K)*Nmax),marker='o')
# #ax.plot(N,np.abs(dNdt_1))
# fig.tight_layout()
# plt.show()

#Solution to problem (iv)

# x = Symbol('x')
# y = Function('y')
# solns = ode.dsolve(Derivative(y(x),x)-y(x)*(1-y(x)**2),y(x))
# soln1 = solns[1]
# print(soln1)

#solution to problem (vi)
def N_ana(t,K,r0,N0,nu):
    if nu==2:
        C1 = 1 - (K/N0)**2
        eterm = np.exp(2*r0*t)
        N = K*np.sqrt(eterm/(eterm-C1))
    elif nu==1: #usual logistic
        C1 = (K/N0) -1
        N = K/(C1*np.exp(-r0*t) + 1)
    return N
t = np.geomspace(1e-2,1e4,1000)
r0 = 0.0346
K = 1000
N0 = [1e1,1e2,1e3,1e4,1e5]
clrs = ['k','r','g','b','m']
fig = plt.figure(figsize=(10,5))
ax = plt.axes()
ax.set_xscale('log')
ax.set_xlabel('Reduced time $r_0 t$')
ax.set_yscale('log')
ax.set_ylabel('$N(t)/K$')
for j, n0 in enumerate(N0):
    clr = clrs[j]
    N = N_ana(t,K,r0,n0,nu=2)
    plt.plot(r0*t,N/K,ls='-',color=clr,label=str(int(n0)))

for j, n0 in enumerate(N0):
    clr = clrs[j]
    N = N_ana(t,K,r0,n0,nu=1)
    plt.plot(r0*t,N/K,ls='--',color=clr,label=str(int(n0)))

plt.legend(title='$N_0$ \n Modified logistic Normal logistic ',loc=4,ncol=2,bbox_to_anchor=(1, 0.6))
plt.title('Convergence to optimal population size')
plt.show()