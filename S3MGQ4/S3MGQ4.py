import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.integrate import odeint

alpha = 1000
alpha0 = 1
beta = 5
n=2

# construction deriv vector
def deriv(z,t):
    m1,m2,m3,p1,p2,p3 = z
    dm1 = -m1 + alpha/(1+p3**n) + alpha0
    dm2 = -m2 + alpha/(1+p1**n) + alpha0
    dm3 = -m3 + alpha/(1+p2**n) + alpha0
    dp1 = -beta*(p1-m1)
    dp2 = -beta*(p2-m2)
    dp3 = -beta*(p3-m3)
    return [dm1,dm2,dm3,dp1,dp2,dp3]

tmax = 100
t = np.linspace(0, tmax, 2000)
z0 = [0,1,0,2,0,5]
z = odeint(deriv, z0, t)
fig, axs = plt.subplots(2,1,sharex=True)
fig.set_size_inches(6,3)
ax = axs[0]
ax.set_xlim((0,1.5*tmax))
ax.set_xlabel('Time')
ax.set_ylabel('mRNA')
colors = ['r','g','b','r','g','b']
lss = ['-','-','-','--','--','--']
for i in range(3):
    ax.plot(t,z[:,i],color=colors[i],ls=lss[i],label='m' + str(i+1))
ax.legend()
ax = axs[1]
ax.set_ylabel('Proteins')
for i in range(3):
    ax.plot(t,z[:,i+3],color=colors[i+3],ls=lss[i+3],label='p' + str(i+1))
ax.legend()
fig.tight_layout()
plt.show()

fig = plt.figure()
fig.set_size_inches(3,3)
ax = plt.axes()
ax.set_xlabel('mRNA concentrations')
ax.set_ylabel('protein concentrations')
for i in range(3):
    ax.plot(z[:,i],z[:,i+3],color=colors[i],label=str(i+1),alpha=0.3)
plt.legend()
plt.show()