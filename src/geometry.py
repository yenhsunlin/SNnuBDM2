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
    root = np.sqrt(2*(Rstar**2*np.cos(theta)**2*(2*D**2 - Rstar**2 + Rstar**2*np.cos(2*theta))))
    common = D**2 + Rstar**2 - 2*Rstar**2*np.sin(theta)**2
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
    d2 = d(D,Rstar,theta,True)
    ell2 = Re**2 + d2*np.cos(theta)**2 - 2*Re*np.sqrt(d2)*np.cos(theta)*np.cos(beta)
    if is_square is False:
        return np.sqrt(ell2)
    else:
        return ell2


# %% Calculate r'
def rPrime(D,Rstar,Re,theta,phi,beta):
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
    ell2 = ell(Re,theta,beta,True)
    # d^2
    d2 = d(D,Rstar,theta,True)
    # h
    h = np.sqrt(d2)*np.sin(theta)
    # cos(iota) and iota
    cosIota = (Re**2 - ell2 - d2*np.cos(theta)**2)/(2*np.cos(theta)*np.sqrt(ell2*d2))
    iota = np.arccos(cosIota)
    # r'^2
    rp2 = ell2*cosIota**2 + (np.sqrt(ell2)*np.sin(iota) - h*np.sin(phi))**2 + h**2*np.cos(phi)**2
    return np.sqrt(rp2)

    

if __name__ == '__main__':
   D =3
   Rstar = 8.5
   Re = 8.5
   theta = 0.01*np.pi
   phi = 0.01
   beta = 0
   print(d(D,Rstar,theta))
   print(sanityCheck(d(D,Rstar,theta), D, theta))
   print(rPrime(D,Rstar,Re,theta,phi,beta))