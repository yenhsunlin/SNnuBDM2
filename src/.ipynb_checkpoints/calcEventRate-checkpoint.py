import os
import numpy as np
import multiprocessing as mp
from functools import partial
import vegas
from constants import *
from supernovaNuBoostedDM import *


# %% ------ Basic inputs ------ %% #
# Path for saving data
savePath = os.getcwd() + '/'
# How many cpu cores to be used
cpus = mp.cpu_count() - 2
# How many iterations for vegas
nitn = 10
# How many evaluation numbers for vegas
neval = 100
# The ratio r_m = mV/mx
r_m = 1/3
# Maximum exposure time
texpMax = 35*yr2s
# Min and max Tx in the integration
Tx_min = 5
Tx_max = 100

# %% Calculate BDM flux vs. time
def eventRatePerElectron(mx,Rstar,beta,Re=8.5,t_cut=texpMax,r_cut=1e-05,gV=1,gD=1,eps=1,tau=10):
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
        return diffEventRateAtDetector(t,Tx,mx,mV,Rstar,theta,phi,beta,Re,r_cut,gV,gD,eps,tau)
    
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
    mx_list = np.logspace(-6,1.5,20)
    Rs_list = [5,8.5,13]
    beta_list = [0,0.5,1] # unit of pi
    gV = 1e-5
    gD = 1e-5
    eps = 1e-5
    
    # Initializing pooling
    pool = mp.Pool(cpus)
    i = 1
    for Rstar in Rs_list:
        for beta in beta_list:
            Beta = beta*np.pi
            targetFuncForParallelization = partial(eventRatePerElectron,Rstar=Rstar,beta=Beta)
            eventRate = np.array(pool.map(targetFuncForParallelization,mx_list))
            eventRate = np.vstack((mx_list,eventRate.T))
            np.savetxt(savePath + f'event_Rs{Rstar:.2f}_beta{beta:.2f}.txt',
                       eventRate.T,fmt='%.5e  %.5e  %d',header='mx         event         exit_code')
            print(f'{i} out of 9 runs are completed',end='\r')
            i+=1

    
    
    #for mx in mx_list:
    #    mV = mx/3
    #    for Rstar in Rs_list:
    #        for Tx in Tx_list:
    #            for beta in beta_list:
    #                # get tvan and theta_M
    #                tvan,thetaMax = get_tvan_thetaM(Tx,mx,Rstar)
    #                Beta = beta*np.pi
    #                # list of time
    #                time_list = np.logspace(np.log10(initial_time),np.log10(tvan),200)
    #                targetFunc = partial(diffFluxAtEarthVersusTime,Tx=Tx,mx=mx,mV=mV,Rstar=Rstar,thetaMax=thetaMax,beta=Beta,gV=gV,gD=gD)
    #                flux = np.array(pool.map(targetFunc,time_list))
    #                flux = np.vstack((time_list,flux.T)) 
    #                #attributes = np.array([Tx,mV,Rstar,beta])
    #                np.savetxt(savePath + f'flux_mx{mx:.2e}_Tx{Tx:0{3}d}_Rs{Rstar:.2f}_beta{beta:.2f}.txt',
    #                           flux.T,fmt='%.5e  %.5e  %d',header='time       flux         exit_code')
                    #print(f'{i} out of {totalRuns} runs are completed',end='\r')
                    #i+=1
        
    #print('All process are completed!',end='\n')