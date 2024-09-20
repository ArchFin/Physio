import numpy as np
import math
from matplotlib import pyplot as plt

NNS = 4.8e6
def repression(R,NNS,DErd):
    return 1 + (R/NNS)*np.exp(-DErd)
fig = plt.figure(figsize=(2,2))
ax = plt.axes()
ax.set_xscale('log')
ax.set_xlabel(r'$R$')
ax.set_yscale('log')
ax.set_ylabel('Repression')
# O1
R = np.array([50,900])
rep = np.array([200,4700])
DErd_O1 = -np.log((NNS/R)*(rep-1))
DErd_O1 = np.mean(DErd_O1)
print("O1: " + "{:.2f}".format(DErd_O1) + " kT")
# generate X, Y data for fitting plot
Rplt = np.geomspace(1,1e4,100)
Repplt = repression(Rplt,NNS,DErd_O1)
ax.plot(R,rep,marker='s',ls='',color='g') # plot true data
ax.plot(Rplt,Repplt,ls='-',color='g') # plot fitting
# O2
R = np.array([50,900])
rep = np.array([21,320])
DErd_O2 = -np.log((NNS/R)*(rep-1))
DErd_O2 = np.mean(DErd_O2)
print("O2: " + "{:.2f}".format(DErd_O2) + " kT")
# generate X, Y data for fitting plot
Rplt = np.geomspace(1,1e4,100)
Repplt = repression(Rplt,NNS,DErd_O2)
ax.plot(R,rep,marker='s',ls='',color='r') # plot true data
ax.plot(Rplt,Repplt,ls='-',color='r') # plot fitting
# O3
R = np.array([50,900])
rep = np.array([1.3,16])
DErd_O3 = -np.log((NNS/R)*(rep-1))
DErd_O3 = np.mean(DErd_O3)
print("O3: " + "{:.2f}".format(DErd_O3) + " kT")
# generate X, Y data for fitting plot
Rplt = np.geomspace(1,1e4,100)
Repplt = repression(Rplt,NNS,DErd_O3)
ax.plot(R,rep,marker='s',ls='',color='b') # plot true data
ax.plot(Rplt,Repplt,ls='-',color='b') # plot fitting
plt.show()