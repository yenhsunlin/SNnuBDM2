# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from constants import *


# %% ---------- Functions for evaluating geometrical relations ---------- %% #

# %% Get the scattering angle alpha
def getCosPsi(d,Rstar,theta):
    """
    Get the cosine value of scattering angle cos(psi).
    If we did it with law of cosine, then for the case of psi > pi/2,
    it will always return pi - psi which cannot reflect the pratical
    situation
    
    Input
    ------
    d: the l.o.s distance d
    Rstar: the distance between Earth and SN
    theta: the open-angle in rad
    
    Output
    ------
    psi: scattering angle in rad
    """
    # Get D^2
    D2 = getD(d,Rstar,theta,True)
    D = np.sqrt(D2)
    # Get cos(alpha)
    cosPsi = (Rstar**2 - D2 - d**2)/(2*D*d)
    if cosPsi > 1: cosPsi = 1
    elif cosPsi < -1: cosPsi = -1
    else: pass
    return cosPsi


# %% Calculate D
def getD(d,Rstar,theta,is_square = False):
    """
    Calculate the distance between SN and boosted point D
    
    Input
    ------
    d: the l.o.s distance d
    Rstar: the distance between Earth and SN
    theta: the open-angle in rad
    is_square: return the square of such distance, default is False
    
    Output
    ------
    D: the distance D
    """
    # Calculate D^2 via law of cosine
    D2 = d**2 + Rstar**2 - 2*d*Rstar*np.cos(theta)
    if is_square is False:
        return np.sqrt(D2)
    elif is_square is True:
        return D2
    else:
        raise ValueError('\'is_square\' must be either True or False')
    

# %% Calculate ell
def getEll(d,Re,theta,beta,is_square = False):
    """
    Calculate the distance ell
    
    Input
    ------
    d: the l.o.s distance d
    Re: the distance between Earth and the GC
    theta: the open-angle in rad
    beta: the off-center angle in rad
    is_square: return the square of such distance, default is False
    
    Output
    ------
    ell: the distance ell
    """
    # Calculate ell^2 via law of cosine
    ell2 = Re**2 + (d*np.cos(theta))**2 - 2*Re*d*np.cos(theta)*np.cos(beta)
    if is_square is False:
        return np.sqrt(ell2)
    elif is_square is True:
        return ell2
    else:
        raise ValueError('\'is_square\' must be either True or False')


# %% Calculate r'
def getRprime(d,Rstar,Re,theta,phi,beta,tolerance = 1e-10):
    """
    Calculate the distance from boosted point to GC r'
    
    Input
    ------
    d: the l.o.s distance d
    Rstar: the distance between Earth and SN
    Re: the distance between Earth and the GC
    theta: the open-angle in rad
    phi: the azimuth angle in rad
    beta: the off-center angle in rad
    
    Output
    ------
    r': the distance r'
    """
    # ell^2
    ell2 = getEll(d,Re,theta,beta,True)
    # D^2
    D2 = getD(d,Rstar,theta,True)
    # h
    h = d*np.sin(theta)
    
    # Calculate cos(iota) and iota
    cosIota = (Re**2 - ell2 - (d*np.cos(theta))**2)/(2*np.cos(theta)*np.sqrt(ell2)*d)
    # Using sin(arccos(x)) = sqrt(1-x^2)
    if cosIota > 1:
        cosIota = 1
    elif cosIota < -1:
        cosIota = -1
    else:
        pass
    sinIota = np.sqrt(1 - cosIota**2)
    
    # Calculate r'^2
    rp2 = ell2*cosIota**2 + (np.sqrt(ell2)*sinIota - h*np.sin(phi))**2 + h**2*np.cos(phi)**2
    return np.sqrt(rp2)


# %% Calculate l.o.s d for a given time
def getd(t,vx,Rstar,theta):
    """
    Calculate the distance l.o.s d
    
    Input
    ------
    t: the arrival time of BDM at Earth relative to the first SN neutrino on the Earth
    vx: BDM velocity in the unit of light speed
    Rstar: the distance between Earth and SN
    theta: the open-angle in rad
    
    Output
    ------
    d: the l.o.s
    """
    zeta = Rstar + lightSpeed*t/kpc2cm
    cosTheta = np.cos(theta)
    d = (zeta - Rstar*vx*cosTheta - np.sqrt((Rstar**2 - zeta**2)*(1 - vx**2) + (Rstar*vx*cosTheta - zeta)**2))*vx/(1-vx**2)
    return d


# %% ---------- Functions for evaluating kinematical relations ---------- %% #

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


# %% Vanishing time
def get_tvan_thetaM(Tx,mx,Rstar):
    """
    Get the vanishing time and maximum theta
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: Mass of DM
    Rstar: Distance from SN to the Earth, kpc
    
    Output
    ------
    tup: maximum theta, rad
    """
    # Get maximum psi and BDM velocity
    psiM = maxPsi(Tx,mx)
    vx = getVelocity(Tx,mx)
    
    # Solving the corresponding maximum theta
    def _thetaM(theta):
        """ Target function """
        return np.cos(psiM - theta)/np.cos(theta) - vx
    thetaM = root_scalar(_thetaM, method='brentq', bracket=[0,np.pi/2]).root

    # Evaluating the vanishing time
    t0 = Rstar*kpc2cm/lightSpeed
    tVan = ((np.sin(thetaM) + np.sin(psiM - thetaM)/vx)/np.sin(psiM) - 1)*t0
    return tVan,thetaM


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