import supernovaNuBoostedDM as s1
import SNNuBDM as s2
import numpy as np
import matplotlib.pyplot as plt

mx = [1e-6, 1e-3, 1]
Rstar = 8.5
theta = 1e-4
Tx = 10
t = np.logspace(-8,4,num=1000, base=10)

d = np.zeros((6,1000))
for i in range(3):
    vx = s2.getVelocity(Tx, mx[i])
    for j in range(1000):
        d[i,j] = s1.getd(t[j], vx, Rstar, theta)
for i in range(3):
    vx = s2.getVelocity(Tx, mx[i])
    for j in range(1000):
        d[i+3,j] = s2.getd(t[j], vx, Tx, mx[i], Rstar, theta)

Fig = plt.figure()
plt.plot(t, d[3, :], color='r', linestyle='-', label=r'$R_{star}=8.5\ kpc\ \ mx=1\ eV\ \ Tx=10\ MeV\ \theta=10^{-4}\ (new\ ver.)$')
plt.plot(t, d[4, :], color='g', linestyle='-', label=r'$R_{star}=8.5\ kpc\ \ mx=1\ keV\ \ Tx=10\ MeV\ \theta=10^{-4}\ (new\ ver.)$')
plt.plot(t, d[5, :], color='b', linestyle='-', label=r'$R_{star}=8.5\ kpc\ \ mx=1\ MeV\ \ Tx=10\ MeV\ \theta=10^{-4}\ (new\ ver.)$')
plt.plot(t, d[0, :], color='r', linestyle='-.', label=r'$R_{star}=8.5\ kpc\ \ mx=1\ eV\ \ Tx=10\ MeV\ \theta=10^{-4}\ (old\ ver.)$')
plt.plot(t, d[1, :], color='g', linestyle='-.', label=r'$R_{star}=8.5\ kpc\ \ mx=1\ keV\ \ Tx=10\ MeV\ \theta=10^{-4}\ (old\ ver.)$')
plt.plot(t, d[2, :], color='b', linestyle='-.', label=r'$R_{star}=8.5\ kpc\ \ mx=1\ MeV\ \ Tx=10\ MeV\ \theta=10^{-4}\ (old\ ver.)$')

plt.legend(loc='best')
plt.title('Difference between Old and New ver. of getd',fontsize=30)
plt.xscale("log", base=10)
plt.xlim(1e-8, 1e4)
plt.yscale("log", base=10)
plt.ylim(1e-20, 1e2)
plt.tick_params(labelsize=24)
plt.legend(fontsize=16)
plt.grid(alpha=0.5)
Fig.set_size_inches(16, 9)
plt.xlabel(r'$time\ [yr]$', fontsize=24)
plt.ylabel(r'$d\ [kpc]$', fontsize=24)
plt.tight_layout()
plt.savefig('test_getd_2.png')
plt.savefig('test_getd_2.jpg')
plt.show()