import numpy as np 
import math 
import matplotlib.pyplot as plt

#This is code to plot the null-clines
Vr = 70
betaq = 0.10
gamma = 0.10
pmin = 1e-4
pmax = 1e0


p = np.geomspace(pmin,pmax,200)
V1 = Vr*p/(p+gamma)
V2 = Vr - (1/betaq)*np.log((1-p)/p)
fig = plt.figure()
ax = plt.axes()
# ax.set_xscale('log')
ax.set_xlim((-5,100))
ax.set_xlabel('V (mV)')
ax.set_yscale('log')
ax.set_ylim((pmin,pmax))
ax.set_ylabel('p')
ax.plot(V1,p,label='V1')
ax.plot(V2,p,label='V2')
plt.legend()
plt.show()

#for vr = 150

Vr = 150

p = np.geomspace(pmin,pmax,200)
V1 = Vr*p/(p+gamma)
V2 = Vr - (1/betaq)*np.log((1-p)/p)
fig = plt.figure()
ax = plt.axes()
# ax.set_xscale('log')
ax.set_xlim((-5,100))
ax.set_xlabel('V (mV)')
ax.set_yscale('log')
ax.set_ylim((pmin,pmax))
ax.set_ylabel('p')
ax.plot(V1,p,label='V1')
ax.plot(V2,p,label='V2')
plt.legend()
plt.show()