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


# %% Sanity check
def sanityCheck(d,D,theta):
    """
    Do a sanity check that the inputs satisfy -1 <= d*sin(theta)/D <= 1
    
    Input
    ------
    d: the l.o.s distance d
    D: the SNv propagation distance D
    theta: the open-angle in rad
    
    Output
    ------
    bool: is the inputs sane?
    """
    return -1 <= d*np.sin(theta)/D <= 1
    

# %% Calculate the l.o.s distance d
def d(D,Rstar,theta,is_square = False):
    """
    Calculate the l.o.s distance d 
    
    Input
    ------
    D: the SNv propagation distance D
    Rstar: the distance between Earth and SN
    theta: the open-angle in rad
    is_square: return the square of such distance, default is False
    
    Output
    ------
    d: the l.o.s distance d
    """
    sinTheta = np.sin(theta)
    root = 2*Rstar*np.sqrt((D + Rstar*sinTheta)*(D - Rstar*sinTheta)*(1 - sinTheta)*(1 + sinTheta))
    #np.sqrt(2*(Rstar**2*np.cos(theta)**2*(2*D**2 - Rstar**2 + Rstar**2*np.cos(2*theta))))
    common = D**2 + Rstar**2 - 2*Rstar**2*sinTheta**2
    # d square, there are two solutions, one with plus sign the other minus sign
    # according to preliminary numerical examination, it seems minus is more
    # plausible. 
    # d2Plus = common + root
    d2Minus = common - root
    if is_square is False:
        return np.sqrt(d2Minus)
    else:
        return d2Minus
    
    
# %% Calculate ell
def ell(Rstar,Re,theta,beta,is_square = False):
    """
    Calculate the distance ell
    
    Input
    ------
    Rstar: the distance between Earth and SN
    Re: the distance between Earth and the GC
    theta: the open-angle in rad
    beta: the off-center angle in rad
    is_square: return the square of such distance, default is False
    
    Output
    ------
    ell: the distance ell
    """
    # Calculate d^2
    d2 = d(D,Rstar,theta,True)
    # Get d
    small_d = np.sqrt(d2)
    # Calculate ell^2
    ell2 = Re**2 + d2*np.cos(theta)**2 - 2*Re*small_d*np.cos(theta)*np.cos(beta)
    if is_square is False:
        return np.sqrt(ell2)
    else:
        return ell2


# %% Calculate r'
def rPrime(D,Rstar,Re,theta,phi,beta,tolerance = 1e-15):
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
    ell2 = ell(Rstar,Re,theta,beta,True)
    # d^2
    d2 = d(D,Rstar,theta,True)
    # h
    h = np.sqrt(d2)*np.sin(theta)
    # cos(iota) and iota
    cosIota = (Re**2 - ell2 - d2*np.cos(theta)**2)/(2*np.cos(theta)*np.sqrt(ell2*d2))
    # Using sin(arccos(x)) = sqrt(1-x^2)
    if 0 <= np.abs(cosIota) <= 1:
        # normal case
        sinIota = np.sqrt(1 - cosIota**2)
    elif np.abs(cosIota) - 1 < tolerance:
        # cosIota is outside the valid range but its value is still within the
        # tolerance, which implies abs(cosIota) = 1 is still ok to our calculation.
        cosIota = 1
        sinIota = 0
    else:
        # cosIota is not only outside the valid range but also untolerable
        raise print('cosIota value is outside the tolarance range, please check again')
    # r'^2
    rp2 = ell2*cosIota**2 + (np.sqrt(ell2)*sinIota - h*np.sin(phi))**2 + h**2*np.cos(phi)**2
    return np.sqrt(rp2),cosIota

    

if __name__ == '__main__':
   D = 11
   Rstar = 8.5
   Re = 8.5
   theta = 0.01*np.pi
   phi = 1
   beta = 0
   print(d(D,Rstar,theta))
   print(sanityCheck(d(D,Rstar,theta),D,theta))
   print(ell(Rstar,Re,theta,beta))
   print((Re- d(D,Rstar,theta)*np.cos(theta)))
   print(rPrime(D,Rstar,Re,theta,phi,beta))