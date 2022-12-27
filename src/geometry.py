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
from constants import *
import warnings


# %% User-derined warning
class ToleranceWarning(UserWarning):
    pass


# %% Get the scattering angle alpha
def get_cosPsi(d,Rstar,theta):
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
    D2 = get_D(d,Rstar,theta,True)
    D = np.sqrt(D2)
    # Get cos(alpha)
    cosPsi = (Rstar**2 - D2 - d**2)/(2*D*d)
    return cosPsi


# %% Calculate D
def get_D(d,Rstar,theta,is_square = False):
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
def get_ell(d,Re,theta,beta,is_square = False):
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
def get_rprime(d,Rstar,Re,theta,phi,beta,tolerance = 1e-10):
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
    ell2 = get_ell(d,Re,theta,beta,True)
    # D^2
    D2 = get_D(d,Rstar,theta,True)
    # h
    h = d*np.sin(theta)
    # cos(iota) and iota
    cosIota = (Re**2 - ell2 - (d*np.cos(theta))**2)/(2*np.cos(theta)*np.sqrt(ell2)*d)
    # Using sin(arccos(x)) = sqrt(1-x^2)
    if 0 <= np.abs(cosIota) <= 1:
        # normal case
        sinIota = np.sqrt(1 - cosIota**2)
    elif Re == Rstar and beta == 0:
        # This is GC = SN case and directly applies cos(iota) = 1
        cosIota = 1
        sinIota = 0
    elif np.abs(cosIota) - 1 <= tolerance:
        # This is not GC = SN case but would be very close.
        # We firstly check if the cos(iota) is within the tolerance range.
        # If so, then we consider it is GC ~ SN
        cosIota = 1
        sinIota = 0
    else:
        # cos(iota) is outside the tolerance range
        warnings.warn('The inputs resulted cos(iota) outsides the tolerance range.', ToleranceWarning)
        sinIota = np.sqrt(cosIota**2 - 1)
    # r'^2
    rp2 = ell2*cosIota**2 + (np.sqrt(ell2)*sinIota - h*np.sin(phi))**2 + h**2*np.cos(phi)**2
    return np.sqrt(rp2)


# %% Calculate l.o.s d for a given time
def get_d(t,vx,Rstar,theta):
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
    zeta = Rstar + light_speed*t/kpc2cm
    cosTheta = np.cos(theta)
    d = (zeta - Rstar*vx*cosTheta - np.sqrt((Rstar**2 - zeta**2)*(1 - vx**2) + (Rstar*vx*cosTheta - zeta)**2))*vx/(1-vx**2)
    return d
