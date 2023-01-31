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
    # Get cos(psi)
    denominator = 2*D*d # check if the denominator in the law of cosine is not 0.0
    if denominator != 0.0:
        numerator = Rstar**2 - D2 - d**2
        cosPsi = numerator/denominator
        # Dealing with round-off error
        if cosPsi > 1: cosPsi = 1
        elif cosPsi < -1: cosPsi = -1
        else: pass
    else:
        # the denominator is 0.0, which means d = 0, applying L'Hospital's rule to get cos(psi)
        cosPsi = 0.0
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
    # D2 might turn minus due to round-off error, it shoud truncate at 0
    if D2 < 0: D2 = 0
    else: pass
    
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
    # ell2 might turn minus due to round-off error, it should truncate at 0
    if ell2 < 0: ell2 = 0.0
    else: pass
    
    if is_square is False:
        return np.sqrt(ell2)
    elif is_square is True:
        return ell2
    else:
        raise ValueError('\'is_square\' must be either True or False')


# %% Calculate r'
def getRprime(d,Re,theta,phi,beta):
    """
    Calculate the distance from boosted point to GC r'
    
    Input
    ------
    d: the l.o.s distance d
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
    # h
    h = d*np.sin(theta)
    
    # Calculate cos(iota) and iota
    denominator = 2*np.cos(theta)*np.sqrt(ell2)*d # check if the denomator in the law of cosine is not 0.0
    if denominator != 0.0:
        numerator = Re**2 - ell2 - (d*np.cos(theta))**2
        cosIota = numerator/denominator
        # Dealing with round-off error
        if cosIota > 1: cosIota = 1
        elif cosIota < -1: cosIota = -1
        else: pass
    else:
        # the denominator is 0, which means d = 0, applying L'Hospital to get cos(iota)
        cosIota = 0
    # Using sin(arccos(x)) = sqrt(1-x^2)
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
    denominator = 1-vx**2
    if denominator != 0.0:
        numerator = (zeta - Rstar*vx*cosTheta - np.sqrt((Rstar**2 - zeta**2)*(1 - vx**2) + (Rstar*vx*cosTheta - zeta)**2))*vx
        return numerator/denominator
    else:
        return 0.0


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


# %% ---------- Functions for evaluating BDM flux and event ---------- %% #

# %% Neutrino number density at distance D
def dnv(D,Ev,Lv = Lv,tau = 10):
    """
    Neutrino number density per energy at D
    
    Input
    ------
    Enu: Neutrino energy in MeV
    D: Distance from the boosted point to the SN explosion site, in kpc
    Lv: Neutrino luminosity, default is 1e52 erg/s
    tau: duration of the SN explosion
    
    Output
    ------
    Neutrino flux at d: # per Enu per cm**3
    """
    Lv = Lv*erg2MeV*tau
    D = D*kpc2cm
    
    # Fermi dirac distribution
    def _fv(Ev,Tv):
        exponent = Ev/Tv - 3
        if exponent <= 700:
            return (1/18.9686)*Tv**(-3)*(Ev**2/(np.exp(exponent) + 1))
        else:
            return 0
    nue_dist = _fv(Ev,2.76)/11
    nueb_dist = _fv(Ev,4.01)/16
    # total 4 species for x
    nux_dist = _fv(Ev,6.26)/25
    
    luminosity = Lv/(4*np.pi*D**2*lightSpeed)
    return luminosity*(nue_dist+nueb_dist+4*nux_dist)


# %% NFW DM number density
def nx(r,mx):
    """
    DM halo number density at r in MW
    
    Input
    ------
    r: distance to GC, in kpc
    mx: DM mass in MeV
    
    Output
    ------
    DM number density, #/cm^3 at r
    """
    rr=r/24.42
    return (184/mx)/(rr*(1 + rr)**2)


# %% BDM emissivity at the direction of psi
def getJx(Tx,mx,mV,r,D,psi,gV=1,gD=1,tau=10):
    """
    Evaluate the BDM emissivity toward the direction psi at the given boosted point 
    
    Input
    ------
    Tx: BDM kinetic energy, MeV
    mx: DM mass, MeV
    mV: mediator mass, MeV
    r: distance from boosted point to GC for calculating DM number density, kpc
    D: distance from boosted point to the SN explosion site, kpc
    psi: the BDM scattering angle, rad
    gV: DM-neutrino coupling constant, default 1
    gD: DM-DM coupling constant, default 1
    
    Output
    ------
    jx: BDM emissivity at the boosted point, 1/(MeV*cm^3*s*rad)
    """   
    # Get the required Ev
    Ev = getEv(Tx,mx,psi)
    # Get dEv/dTx
    dEvdTx = dEv(Tx,mx,psi) 
    # Get the differential DM-nu scattering cross section
    diffCrox = diffCrossSectionNuDM(Tx,mx,mV,psi,gV,gD)
    # Get the emissivity jx
    jx = lightSpeed*diffCrox*nx(r,mx)*dnv(D,Ev,Lv,tau)*dEvdTx
    return jx


# %% Differential BDM flux at Earth
def diffFluxAtEarth(t,Tx,mx,mV,Rstar,theta,phi,beta,Re=8.5,r_cut=1e-5,gV=1,gD=1,tau=10):
    """
    The differential BDM flux over open angle theta at Earth
    
    Input
    ------
    t: The differential BDM flux at time t, relative to the first SN neutrino
        arriving at Earth
    Tx: BDM kinetic energy, MeV
    mx: DM mass, MeV
    mV: Mediator mass, MeV
    Rstar: Distance from Earth to SN, kpc
    theta: The open angle theta
    phi: The azimuthal angle along the Earth-SN axis, rad
    beta: The deviation angle, characterizing how SN deviates the GC, rad
    Re: The distance from Earth to GC, default 8.5 kpc
    r_cut: Ignore the BDM contribution when r' < r_cut, default 1e-5 kpc
    gV: DM-neutrino coupling constant, default 1
    gD: DM-DM coupling constant, default 1
    tau: The duration of SN explosion, default 10 s
    
    Output
    ------
    scalar: The diff. BDM flux at Earth, # per MeV per cm^2 per second per sr
    """
    # Get BDM velocity
    vx = getVelocity(Tx,mx)
    # Get the propagation length of BDM via given t and vx
    d = getd(t,vx,Rstar,theta)
    # Get the required SNv propagation length
    D = getD(d,Rstar,theta)
    # Get the distance between boosted point to GC
    rprime = getRprime(d,Re,theta,phi,beta)
    if  D != 0.0 and ~np.isnan(rprime) and rprime >= r_cut:
        # Get the BDM scattering angle psi
        psi = np.arccos(getCosPsi(d,Rstar,theta))
        # Evaluate the xemissivity
        jx = getJx(Tx,mx,mV,rprime,D,psi,gV,gD,tau)
        # Jacobian
        J = lightSpeed/((d - Rstar*np.cos(theta))/D + 1/vx)
        # BDM flux
        return J*jx*vx*np.sin(theta)
    else:
        return 0


# %% Differential BDM event rate in the detector
def diffEventRateAtDetector(t,Tx,mx,mV,Rstar,theta,phi,beta,Re=8.5,r_cut=1e-5,gV=1,gD=1,eps=1,tau=10):
    """
    Calculate the differential event rate at the detector, the unit is:
    # per second per MeV per sr per electron
    To obtain the total event one should integrate this function over time, Tx,
    steradian and electron number in the given detector
    
    Input
    ------
    t: The differential BDM flux at time t, relative to the first SN neutrino
        arriving at Earth
    Tx: BDM kinetic energy, MeV
    mx: DM mass, MeV
    mV: Mediator mass, MeV
    Rstar: Distance from Earth to SN, kpc
    theta: The open angle theta
    phi: The azimuthal angle along the Earth-SN axis, rad
    beta: The deviation angle, characterizing how SN deviates the GC, rad
    Re: The distance from Earth to GC, default 8.5 kpc
    r_cut: Ignore the BDM contribution when r' < r_cut, default 1e-5 kpc
    gV: DM-neutrino coupling constant, default 1
    gD: DM-DM coupling constant, default 1
    eps: DM-electron coupling constant, default 1
    tau: The duration of SN explosion, default 10 s
    
    Output
    ------
    scalar: The differential event rate
    """
    # The BDM flux at Earth
    diffFlux = diffFluxAtEarth(t,Tx,mx,mV,Rstar,theta,phi,beta,Re,r_cut,gV,gD,tau)
    # The DM-e cross section
    croxDMe = totalCrossSectionDMe(Tx,mx,mV,eps,gD)
    # The differential event rate
    diffEvent = diffFlux*croxDMe
    return diffEvent