import supernovaNuBoostedDM as s1
import SNNuBDM as s2
import numpy as np
import matplotlib.pyplot as plt

Ell = np.zeros((2,1000))
d = 8.5
Re = 8.5
theta = 0
beta = np.logspace(-14,0,num=1000, base=10)


for i in range(1000):
    Ell[0,i] = s1.getEll(d,Re,theta,beta[i])
for i in range(1000):
    Ell[1,i] = s2.getEll(d,Re,theta,beta[i])

Fig = plt.figure()
plt.plot(beta, Ell[1, :], color='r', linestyle='-', label=r'$R_{star}=8.5\ kpc\ \ d=8.5\ kpc\ \ \theta=0\ (new\ ver.)$')
plt.plot(beta, Ell[0, :], color='b', linestyle='-', label=r'$R_{star}=8.5\ kpc\ \ d=8.5\ kpc\ \ \theta=0\  (old\ ver.)$')
plt.legend(loc='best')
plt.title('Difference between Old and New ver. of getEll',fontsize=30)
plt.xscale("log", base=10)
plt.xlim(1e-10, 1e-4)
plt.yscale("log", base=10)
plt.ylim(1e-10, 1e-4)
plt.tick_params(labelsize=24)
plt.legend(fontsize=16)
plt.grid(alpha=0.5)
Fig.set_size_inches(16, 9)
plt.xlabel(r'$\beta\ [rad]$', fontsize=24)
plt.ylabel(r'$\ell\ [kpc]$', fontsize=24)
plt.tight_layout()
plt.savefig('test_getEll.png')
plt.savefig('test_getEll.jpg')
plt.show()