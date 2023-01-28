# -*- coding: utf-8 -*-
"""
*-------- Rules for placing variables in a function --------*

1. Variable prior to constant
2. Distance prior to angle
3. SNv propagation distance D has lower priority than other variable distance
4. Open-angle (theta) is prior to azimuth angle (phi)
5. Off-center angle (beta) is considered as a constant as it does not subjec to
   change when it is assigned in the beginning

"""

import numpy as np
from scipy.integrate import quad
from constants import *


# %% Scattering amplitude 
def amplitudeSquared(s,t,u,m1,m2,mV):
    """
    The spin-averaged amplitude squared for vector interactio
    
    Input
    ------
    s: Mandelstam variables s
    t: Mandelstam variables t
    u: Mandelstam variables u
    m1: Mass of incoming particle 1
    m2: Mass of particle 2 to be scattered off
    mV: Mediator mass
    
    Output
    ------
    scalar: spin-averaged amplitude squared
    """
    # Define sum over mass squared
    massSum = m1**2 + m2**2
    # Spin-averaged amplitude-squared
    ampSq = 2/(t - mV**2)**2*(s**2 + u**2 + 4*t*massSum - 2*massSum**2)
    return ampSq


# %% Calculate momentum for a particle with mass m and kinetic energy T
def getMomentum(T,m):
    """
    Get the momentum of particle with mass m and kinetic energy T
    
    Input
    ------
    T: Kinetic energy of the particle
    m: Mass of the particle
    
    Output
    ------
    scalar: The associated particle momentum
    """
    return np.sqrt(T*(T + 2*m))


# %% Calculate velocity for a particle with mass m and kinetic energy T
def getVelocity(T,m):
    """
    Get the velocity of particle with mass m and kinetic energy T
    
    Input
    ------
    T: Kinetic energy of the particle
    m: Mass of the particle
    
    Output
    ------
    scalar: The associated dimensionless particle velocity
    """
    return getMomentum(T,m)/(T + m)


# %% Calculate the required Ev to upscatter DM to kinetic energy Tx at angle psi
def getEv(Tx,mx,psi):
    """
    Get the required neutrino energy to boost DM up with kinetic
    energy Tx at angle psi
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    psi: Scattering angle
    
    Output
    ------
    Ev: The requires neutrino energy. If the resulting Ev is negative,
        then it is unphysical
    """
    px = getMomentum(Tx,mx)
    return -mx*Tx/(Tx - px*np.cos(psi))


# %% Calculate dEv/dTx
def dEv(Tx,mx,psi):
    """
    Get the dEv/dTx
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    psi: Scattering angle
    
    Output
    ------
    scalar: dEv/dTx
    """
    px = getMomentum(Tx,mx)
    x = np.cos(psi)
    return mx**2*Tx*x/(Tx - px*x)**2
    

# %% Calculate the s,t and u in lab frame for a particle 2 to be scattered off by massless particle 1
def stuLab(T,m2,psi):
    """
    The Mandelstam variables s, t and u in lab frame for massless
    incoming particle 1 and massive particle 2 at rest in the beginning
    
    Input
    ------
    T: The kinetic energy received by m2 after scattering
    m2: Mass of particle 2 to be scattered off
    psi: The scattering angle of m2 with respect to the incoming particle 1
    
    Output
    ------
    tup: (s,t,u)
    """
    p = getMomentum(T,m2)
    x = np.cos(psi)
    s = m2**2*(p*x + T)/(p*x - T)
    t = -2*m2*T
    u = m2**2 - t*(1 + m2/(T - p*x))
    return s,t,u


# %% Maximum psi allowed for a given DM mass and Tx
def maxPsi(Tx,mx):
    """
    Get the maximumly allowed scattering angle psi
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    
    Output
    ------
    psi_max: maximum psi, in rad
    """
    maxCosValue = np.sqrt(Tx/(Tx + 2*mx))
    return np.arccos(maxCosValue)


# %% Maximum open angle theta allowed for a given DM mass and Tx
def maxtheta(Tx,mx,D,Rstar):
    """
    Get the maximum open angle theta that results in non-zero BDM flux
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    D: Distance from boosted point to the SN explosion site, kpc
    Rstar: Distance from SN to the Earth, kpc
    
    Output
    ------
    theta_max: maximum theta, rad
    """
    psiM = maxPsi(Tx,mx)
    thetaM = np.arcsin(D*np.sin(psiM)/Rstar)
    return thetaM


# %% Calculate the differential cross section for nu-DM scattering
def diffCrossSectionNuDM(Tx,mx,mV,psi,gV,gD):
    """
    Get the differential Nu-DM scattering cross section over psi
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    mV: Mediator mass
    psi: Scattering angle
    gV: The coupling strength for V-Nu-Nu vertex
    gD: The coupling strength for V-DM-DM vertex
    
    Output
    ------
    scalar: differential cross section, cm^2 per rad
    """
    if psi < maxPsi(Tx,mx):
        # Get the Mandelstam variables
        s,t,u = stuLab(Tx,mx,psi)
        # Get the amplitude squared
        ampSq = amplitudeSquared(s,t,u,0,mx,mV)
        # Differential cross section
        diffCrox = (gV*gD)**2*ampSq*np.sqrt((1/mx + 2/Tx)/mx**3)*np.sin(psi)*to_cm2
    else:
        diffCrox = 0
    return diffCrox


# %% Kallen lambda function
def kallenLambda(x,y,z):
    return x**2 + y**2 + z**2 - 2*(x*y + y*z + z*x)


# %% Calculate the total DM-electron scattering cross section
def totalCrossSectionDMe(Tx,mx,mV,eps,gD):
    """
    Get the total DM-electron scattering cross section
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    mV: Mediator mass
    eps: The coupling strength for V-e-e vertex
    gD: The coupling strength for V-DM-DM vertex
    
    Output
    ------
    scalar: total cross section, cm^2
    """
    s = mx**2 + me**2 + 2*(mx + Tx)*me 
    ExSq = (s + mx**2 - me**2)**2/4/s    # Ex^2
    pSq = (s - (mx + me)**2)*(s - (mx - me)**2)/4/s    # p^2
    
    # Define the d\sigma/dt
    def _dsig(t):
        u = 2*(mx**2 + me**2) - s - t
        return to_cm2*amplitudeSquared(s,t,u,mx,me,mV)
    
    # Define the integration range
    tm = 2*mx**2 - 2*(ExSq + pSq)
    tp = 2*mx**2 - 2*(ExSq - pSq)
    # Evaluating the integral \int dt*d\sigma/dt
    totCrox,_ = quad(_dsig,tm,tp)
    return totCrox*(gD*eps)**2**eSquared/64/np.pi/s/pSq