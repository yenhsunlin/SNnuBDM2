# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
    

# %% Calculate d^2
def dSquare(D,Rstar,theta):
    """
    Calculate the l.o.s distance d 
    
    Input
    ------
    D: the SNv propagation distance D
    Rstar: the distance between Earth and SN
    theta: the open-angle in rad
    
    Output
    ------
    d: the l.o.s distance d
    """
    root = np.sqrt(2*(Rstar**2*np.cos(theta)**2*(2*D**2 - Rstar**2 + Rstar**2*np.cos(2*theta))))
    common = D**2 + Rstar**2 - 2*Rstar**2*np.sin(theta)**2
    # d square, two terms
    d2Plus = common + root
    d2Minus = common - root
    return np.sqrt(d2Plus),d2Minus
    
    
# %%
#def 

if __name__ == '__main__':
   print( 1> 5 < 7)
   print(np.sqrt((5,7)))