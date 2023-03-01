# -*- coding: utf-8 -*-
__doc__ = """
          This is the script that contains functions to calculate
          BDM flux from every place in MW with model-agonstic pers-
          pective
          """
__author__ = "Yen-Hsun Lin (yenhsun@phys.ncku.edu.tw)"
__date__ = "20230301"
__version__ = "1.0"
__all__ = ('diffFluxAtEarthLegacy',)


# %% --------------------------------------------------------------------- %% #
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from constants import *
from supernovaNuBoostedDM import getd,getD,getRprime,getEv,getCosPsi,getVelocity,dEv,dnv,nx


# %% Gamma factor in special relativity
def gamma(Ev,mx):
    """
    Calculate gamma factor in CM frame
    """
    s = mx**2+2*Ev*mx
    Ecm = 0.5*(s+mx**2)/np.sqrt(s)
    return Ecm/mx


# %% Angular distribution
def g(Ev,mx,psi):
    """
    Calculate BDM angular distribution dndOmega
    
    Input
    ------
    Enu: Neutrino energy
    mx: DM mass
    psi: lab frame scattering angle in [0,Pi/2]
    
    Output
    ------
    g: PDF of scattering angle alpha
    """ 
    if 0 <= psi <= np.pi/2 and Ev > 0:
        gm = gamma(Ev,mx)
        sec = 1/np.cos(psi)
        dndOmega = gm**2*sec**3/np.pi/(1+gm**2*np.tan(psi)**2)**2
    else:
        dndOmega = 0
    return dndOmega


# %% dEv/dTx
def dEv(Tx,mx,psi):
    """
    Calculate dEv/dTx via analytical expression. Note that the given
    inputs must lead to positive Ev or the corresponding dEvdTx would
    be unphysical
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    psi: scattering angle in lab frame
    
    Output
    ------
    dEv/dTx: the derivative of Ev over Tx
    """
    sec = 1/np.cos(psi)
    numerator = mx**2*sec*(2*sec*np.sqrt(Tx*(2*mx + Tx)) + 2*mx + Tx*sec**2 + Tx)
    denominator = (Tx*np.tan(psi)**2 - 2*mx)**2*np.sqrt(Tx*(2*mx + Tx))
    return numerator/denominator


# %% Get the BDM emissivity
def getJx(Tx,mx,psi,r,D,sigxv=1e-45,tau=10,Lv=Lv):
    """
    BDM emissivity at shell r, note the returned result is divided by sigxv and dimensionless DM velocity
    
    Input
    ------
    Tx: DM kinetic energy, MeV
    mx: DM mass, MeV
    psi: the scattering angle in lab frame
    r: the distance from the scattering point to GC, in kpc
    sigxv: DM-neutrino cross section, cm^2
    tau: the duration of SN explosion, default 10 s
    Lv: neutrino luminosity during the SN explosion, default 1e52 erg/s
        note that the total luminosity released by a single SN explosion
        should be Lv times the duration of that explosion, where the unit
        per second will be eliminated from this procedure
    
    Output
    ------
    BDM emissivity: in the unit of per cm^3 per second
    """
    # if Tx >= 2mx/tan(a)^2, Ev will turn negative (diverge at equal) which is unphysical
    if Tx < 2*mx/np.tan(psi)**2:
        Ev = getEv(Tx, mx, psi)
        dEdT = dEv(Tx,mx,psi)
        jx = dnv(D,Ev,Lv,tau)*g(Ev,mx,psi)*dEdT*lightSpeed*nx(r,mx)*tau*sigxv
    else:
        jx = 0
    return jx


# %% BDM flux at Earth
def diffFluxAtEarthLegacy(t,Tx,mx,Rstar,theta,phi,beta,sigxv=1e-45,Re=8.5,r_cut=1e-5,tau=10):
    """
    The differential BDM flux over open angle theta at Earth
    
    Input
    ------
    t: The differential BDM flux at time t, relative to the first SN neutrino
        arriving at Earth
    Tx: BDM kinetic energy, MeV
    mx: DM mass, MeV
    Rstar: Distance from Earth to SN, kpc
    theta: The open angle theta
    phi: The azimuthal angle along the Earth-SN axis, rad
    beta: The deviation angle, characterizing how SN deviates the GC, rad
    sigxv: DM-neutrino cross section, default 1e-45 cm^2
    Re: The distance from Earth to GC, default 8.5 kpc
    r_cut: Ignore the BDM contribution when r' < r_cut, default 1e-5 kpc
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
        jx = getJx(Tx,mx,psi,rprime,D,sigxv,tau,Lv)
        # Jacobian
        J = lightSpeed/((d - Rstar*np.cos(theta))/D + 1/vx)
        # BDM flux
        return J*jx*vx*np.sin(theta)
    else:
        return 0
    