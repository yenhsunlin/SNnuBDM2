#import os
import numpy as np
import multiprocessing as mp
from functools import partial
import vegas
from constants import *
from supernovaNuBoostedDM import *


# %% ------ Basic inputs ------ %% #
# Path for saving data
savePath = ''
# How many cpu cores to be used
cpus = mp.cpu_count() 
# How many iterations for vegas
nitn = 10
# How many evaluation numbers for vegas
neval = 100000
# The ratio r_m = mV/mx
r_m = 1
# Maximum exposure time
texpMax = 35*yr2s
# Min and max Tx in the integration
Tx_min = 5
Tx_max = 100
# Coupling constants
gV = 1
gD = 1
eps = 1

# %% Calculate BDM flux vs. time
def eventRatePerElectron(mx,Rstar,beta,Re=8.5,t_cut=texpMax,r_cut=1e-05,gV=gV,gD=gD,eps=eps,tau=10):
    # Get mV
    mV = r_m*mx
    # Get vanishing time and thetaMax
    tvan,thetaMax = get_tvan_thetaM(Tx_min,mx,Rstar)
    # Setup maximum exposure time texp
    if tvan > t_cut:
        texp = t_cut
    else:
        texp = tvan
    # Target function
    def _targetFunc(x):
        t = x[0]
        Tx = x[1]
        theta = x[2]
        phi = x[3]
        return diffEventRateAtDetector(t,Tx,mx,mV,Rstar,theta,phi,beta,Re,r_cut,gV,gD,eps,tau)*epsPrime(Tx,gV)
    
    # Time bound, Tx bound, theta bound, phi bound
    integrand = vegas.Integrator([[10,texp],[Tx_min,Tx_max],[0,thetaMax],[0,2*np.pi]])
    try:
        result = integrand(_targetFunc,nitn=nitn,neval=neval)
        result = result.mean
        exit_code = 0 # vegas existed sucessfully
    except:
        result = np.nan
        exit_code = 1 # vegas existed with error
    return result,exit_code


# %% ------ Executable starts here ------ %% #
if __name__ == '__main__':

    # DM mass and Tx to be calculated
    mx_list = np.logspace(-6,1.5,50)
    Rs_list = [5,8.5,13]
    beta_list = [0,0.5,1] # unit of pi
    
    # Initializing pooling
    pool = mp.Pool(cpus)
    #i = 1
    for Rstar in Rs_list:
        for beta in beta_list:
            Beta = beta*np.pi
            targetFuncForParallelization = partial(eventRatePerElectron,Rstar=Rstar,beta=Beta)
            eventRate = np.array(pool.map(targetFuncForParallelization,mx_list))
            eventRate = np.vstack((mx_list,eventRate.T))
            np.savetxt(savePath + f'eventPerElectron_equalMv_noEps_Rs{Rstar:.2f}_beta{beta:.2f}.txt',
                       eventRate.T,fmt='%.5e  %.5e  %d',header='mx         event         exit_code')
            #print(f'{i} out of 9 runs are completed',end='\r')
            #i+=1
