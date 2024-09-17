import numpy as np
from matplotlib import pyplot as plt
import sympy.solvers.ode as ode
from sympy import Function, Derivative, Symbol
from scipy.stats import linregress


pA = 0.9 # initial prob of allele A
npop = 50 # number of members of population
ngens = 100 # number of generations to consider
niter = 100 # number of iterations of the simulation to run to get averages
heteros = np.zeros((niter,ngens))


## Wright fisher model
for i in range(niter):

    s = np.random.choice([0, 1], size=(npop,), p=[1-pA, pA])

    for j in range(ngens):
        pA_new = np.sum(s)/len(s)
        #print(pA_new)
        s_new = np.random.binomial(1, pA_new, npop)
        s = s_new
        H = 2*pA_new*(1-pA_new)
        heteros[i,j] = H

# calculate mean heterozygosity
Hmean = np.mean(heteros,axis=0)
# calculate model WF hetero
x = np.arange(0,ngens,1)
Hfit = 2*0.1*0.9*np.exp(-x/npop)

# plot
fig = plt.figure(figsize=(5,2))
ax = plt.axes()
ax.set_xlim((0,ngens))
ax.set_ylim((-0.1,0.6))
for i in range(niter):
    ax.plot(heteros[i,:],'k',lw=0.2)
ax.plot(Hmean,'r',label='H, sim')
ax.plot(Hfit,'b',label='H, model')
plt.legend()
plt.show()

## Moran model
heteros = np.zeros((niter,ngens))
for i in range(niter):
    s = np.random.choice([0, 1], size=(npop,), p=[1-pA, pA])
    for j in range(ngens):
        # select two indices at random. Assign the second one the same value as the first
        # first one "replicates" while the second one "dies"
        idxs = np.random.choice(len(s), 2)
        s[idxs[1]] = s[idxs[0]]
        pA_new = np.sum(s)/len(s)
        H = 2*pA_new*(1-pA_new)
        heteros[i,j] = H

# calculate mean heterozygosity
Hmean = np.mean(heteros,axis=0)
# calculate model WF hetero
x = np.arange(0,ngens,1)
Hfit = 2*0.1*0.9*np.exp(-2*x/npop**2)
# plot
fig = plt.figure(figsize=(5,2))
ax = plt.axes()
ax.set_xlim((0,ngens))
ax.set_ylim((-0.1,0.6))
for i in range(niter):
    ax.plot(heteros[i,:],'k',lw=0.2)
ax.plot(Hmean,'r',label='H, sim')
ax.plot(Hfit,'b',label='H, model')
plt.legend()
plt.show()

## Random Walk 

for i in range(niter):
    pA = 0.5 # initial prob of allele A
    for j in range(ngens):
        if pA >= 1:
            pA = 1
        elif pA <=0:
            pA=0
        else:
            rand = np.random.randint(2, size=1)[0]

            pA += (2*rand-1)/npop # +- 1/N step

            H = 2*pA*(1-pA)

            heteros[i,j] = H
# calculate mean heterozygosity
Hmean = np.mean(heteros,axis=0)
# calculate model WF hetero
x = np.arange(0,ngens,1)
Hfit = 2*0.5*0.5 - 2*x/npop**2
# plot
fig = plt.figure(figsize=(5,2))
ax = plt.axes()
ax.set_xlim((0,ngens))
ax.set_ylim((-0.1,0.6))
for i in range(niter):
    ax.plot(heteros[i,:],'k',lw=0.2)
ax.plot(Hmean,'r',label='H, sim')
ax.plot(Hfit,'b',label='H, model')
plt.legend()
plt.show()