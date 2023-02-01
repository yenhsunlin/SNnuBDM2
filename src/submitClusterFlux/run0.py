#import os
import numpy as np
import multiprocessing as mp
from functools import partial
import vegas
from constants import *
from supernovaNuBoostedDM import *


# %% ------ Basic inputs ------ %% #
# Path for saving data
savePath = ''#os.getcwd()
# How many cpu cores to be used
cpus = mp.cpu_count()
# How many iterations for vegas
nitn = 10
# How many evaluation numbers for vegas
neval = 5000


# %% Calculate BDM flux vs. time
def diffFluxAtEarthVersusTime(t,Tx,mx,mV,Rstar,thetaMax,beta,Re=8.5,r_cut=1e-05,gV=1,gD=1,tau=10):
    
    def _targetFunc(x):
        theta = x[0]
        phi = x[1]
        return diffFluxAtEarth(t,Tx,mx,mV,Rstar,theta,phi,beta,Re,r_cut,gV,gD,tau)
    
    integrand = vegas.Integrator([[0,thetaMax],[0,2*np.pi]])
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
    mx_list = [1e-6,1e-5]
    Tx_list = [5,10,20,40]
    Rs_list = [3,8.35,8.5,8.65,14]
    beta_list = [0,0.25,0.5,1] # unit of pi
    initial_time = 10 # seconds
    totalRuns = len(mx_list)*len(Tx_list)*len(Rs_list)*len(beta_list)
    gV = 1e-6
    gD = 1e-6
    # 
    pool = mp.Pool(cpus)
    #i = 1
    for mx in mx_list:
        mV = mx/3
        for Rstar in Rs_list:
            for Tx in Tx_list:
                for beta in beta_list:
                    # get tvan and theta_M
                    tvan,thetaMax = get_tvan_thetaM(Tx,mx,Rstar)
                    Beta = beta*np.pi
                    # list of time
                    time_list = np.logspace(np.log10(initial_time),np.log10(tvan),200)
                    targetFunc = partial(diffFluxAtEarthVersusTime,Tx=Tx,mx=mx,mV=mV,Rstar=Rstar,thetaMax=thetaMax,beta=Beta,gV=gV,gD=gD)
                    flux = np.array(pool.map(targetFunc,time_list))
                    flux = np.vstack((time_list,flux.T)) 
                    #attributes = np.array([Tx,mV,Rstar,beta])
                    np.savetxt(savePath + f'flux_mx{mx:.2e}_Tx{Tx:0{3}d}_Rs{Rstar:.2f}_beta{beta:.2f}.txt',
                               flux.T,fmt='%.5e  %.5e  %d',header='time       flux         exit_code')
                    #print(f'{i} out of {totalRuns} runs are completed',end='\r')
                    #i+=1
        
    #print('All process are completed!',end='\n')