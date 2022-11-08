import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar


# ------ Useful constant ------ #
to_cm2 = (1.973e-11)**2  # 1/MeV^2 to cm^2
me = 0.511               # electron mass in MeV
mtau = 1776.86           # tau mass in MeV
mmu = 105.658            # muon mass in MeV
g2MeV = 1/(1.783e-27)    # gram to MeV


# ------ Amplitude and gneral differential cross sections ------ #
def spin_avg_M2_2vector(s,t,m1,m2,mV):
    """
    The spin-averaged amplitude squared for vector interaction on both vertices
    
    Input
    ------
    s: Mandelstam variables s
    t: Mandelstam variables t
    m1: the particle that is going to scatter by particle 1
    m2: the particle that is going to be scattered off
    mV: mediator mass
    
    Output
    ------
    scalar: spin-averaged amplitude squared, dimensionless
    """
    # Define sum over mass squared and u channel
    mass_sum = m1**2+m2**2
    u = 2*mass_sum - s - t
    
    # Spin-averaged amplitude-squared
    sp_avg_M2 = 2/(t-mV**2)**2*(s**2+u**2+4*t*mass_sum-2*mass_sum**2)
    
    return sp_avg_M2


def diff_sig_fi(s,t,m1,m2,mV):
    """
    Frame-independent differential cross section, the Mandelstam variable u is
    determined by the relation 2(mx^2+mf^2)-s-t
    
    Input
    ------
    s: Mandelstam variables s
    t: Mandelstam variables t
    m1: the particle that is going to scatter by particle 1
    m2: the particle that is going to be scattered off
    mV: mediator mass
    
    Output
    ------
    scalar: Frame-independent differential cross section  with unit 1/MeV^3
    """
    #u = 2*(mf**2+mx**2)-s-t
    p_star_squared = (s-(m1+m2)**2)*(s-(m1-m2)**2)/4
    sp_avg_M2 = spin_avg_M2_2vector(s,t,m1,m2,mV)
    diff_sig = sp_avg_M2/(64*np.pi*p_star_squared)
    
    return diff_sig


def diff_sig_lab(Ef,m2,mV,phi):
    """
    Lab-frame differential cross section for massless particle 1
    
    Input
    ------
    Ef: The initial energy of particle f
    m2: the particle that is going to be scattered off by particle 1 
    mV: mediator mass
    phi: The scattering angle of massless particle 1
    
    Output
    ------
    scalar: Lab-frame differential cross section with unit 1/MeV^2 * 1/rad
    """
    # Energy of particle f after scattering
    Ef_p = Ef*m2/(m2+Ef*(1-np.cos(phi)))
    
    # The Mandelstam variables
    s = 2*Ef*m2+m2**2
    t = -2*Ef*Ef_p*(1-np.cos(phi))
    #u = -2*mx*Ef_p+mx**2
    
    sp_avg_M2 = spin_avg_M2_2vector(s,t,0,m2,mV)
    diff_sig = (1/32/np.pi)/(m2+Ef*(1-np.cos(phi)))**2*sp_avg_M2
    
    return diff_sig


# ------ Auxillary functions for kinematics ------ #
def Kallen_lambda(x,y,z):
    """
    The Kallen lambda function
    """
    return x**2+y**2+z**2-2*(x*y+y*z+z*x)


def get_tpm(m1,m2,s):
    """
    Get the integration range for frame-independent cross section
    
    Input
    ------
    m1: the particle that is going to scatter
    m2: the particle that is going to be scattered off by particle 1
    s: the s-channel
    
    Output
    ------
    tuple: the allowd range for t: (t_max,t_min)
    """
    E_star2 = (s + m1**2 - m2**2)**2/4/s
    p_star2 = Kallen_lambda(s,m1**2,m2**2)/4/s
    tp = 2*m1**2 -2*E_star2 + 2*p_star2
    tm = 2*m1**2 -2*E_star2 - 2*p_star2
    #tp = (mx**2+mf**2)/2 - s/2 - (mx**2-mf**2)**2/2/s + Kallen_lambda(s,mf**2,mx**2)/2/s
    #tm = tp - Kallen_lambda(s,mf**2,mx**2)/s
    #tm = (mx**2+mf**2)/2 - s/2 - (mx**2-mf**2)**2/2/s - Kallen_lambda(s,mx**2,mf**2)/2/s
    return tp,tm


def px(Tx,mx):
    """
    Get the BDM momentum, please do check if the inputs meet the
    physical requirements
    """
    return np.sqrt(Tx*(Tx + 2*mx))


def cosphi(Ef,Tx,mx,psi):
    """
    Get the cos\phi, please do check if the inputs meet the physical
    requirements
    """
    return 1/np.sqrt(1 + np.sin(psi)**2/(Ef/px(Tx,mx) - np.cos(psi))**2)


def Efp(Ef,Tx,mx,psi):
    """
    Energy of f particle after scattering, please do check if the
    inputs meet the physical requirements
    """
    return Ef*mx/(mx + Ef*(1 - cosphi(Ef,Tx,mx,psi)))


def get_Ef_phi(Tx,mx,psi,max_Ef = 1000):
    """
    Get the initial f energy and scattering angle phi for a given
    (Tx,mx,psi). Additional \'flag\' and \'msg\' will be output to
    indicate the set of solution is valid or not
    
    Input
    ------
    Tx: Kinetic energy of the BDM
    mx: DM mass
    psi: the DM scattering angle, according to the scheme, it should
    be negative (below the horizon)
    max_Ef: Maximum Ef range for the algorithm to search the solution
    
    Output
    ------
    tup: (Ef,phi,flag,msg)
    Ef: Required initial energy for f particle, MeV
    phi: The scattering angle for f particle, rad
    flag: \'valid\' or \'invalid\' for the solution
    msg: 1: Pass, if flag returns \'valid\'
         2: The algorithm cannot find the solution for E_f for the
            given \'max_Ef\'. Or maybe the inputs lead to unphysical
            situation such as violating energy conservation
         3: px > Ef, violates energy conservation
    """
    # Equation for getting Tx
    def _Tx(Ef):
        return Ef - Efp(Ef,Tx,mx,psi)
    # Target function for root_scalar to find Ef -> Efp - Ef = (input Tx)
    def _f(Ef):
        return _Tx(Ef) - Tx
    # Try to solve the Ef with root_scalar
    try:
        Ef = root_scalar(_f, bracket=[0, max_Ef], method='brentq').root
        # Get phi via arccos instead of arctan to aviod minus phi 
        phi = np.arccos(cosphi(Ef,Tx,mx,psi))
        # Writting down the flag and message
        flag = 'valid'
        msg = 1
    except:
        # Error occured when solve the Ef
        Ef = np.nan
        phi = np.nan
        # Writting down the flag and message
        flag = 'invalid'
        msg = 2
    
    # Check the condition
    if ~np.isnan(Ef):
        # If Ef is not NaN, the energy conservation should be obeyed without doubt,
        # the following check might be redundant
        if Ef/px(Tx,mx) >= 1:
            pass
        else:
            # Violates energy conservation
            flag = 'invalid'
            msg = 3
    else:
        pass
    
    return Ef,phi,flag,msg


# ------ Physical cross sections and kinetic mixing induced by mu/tau loops ------ # 
def eps_prime(gV,q):
    """
    Get the epsilon induced by mu/tau loops
    
    Input
    ------
    gV: the DM-lepton coupling constant
    q: the momentum transfer, in MeV
    
    Output
    ------
    scalar: the epsilon
    """
    # electric charge
    e = np.sqrt(4*np.pi/137)
    
    # Define q^2. In this t-channel, q^2 is always spacelike
    q2 = -q**2
    # define the integrand
    integ = lambda x: x*(1-x)*np.log((mtau**2-x*(1-x)*q2)/(mmu**2-x*(1-x)*q2))*e*gV/2/np.pi**2
    # evaluate epsilon
    eps_p,_ = quad(integ,0,1)
    return -eps_p


def diff_sig_xnu(Tx,mx,mV,psi,gV,gD,max_Ef=1000):
    """
    Differential cross section for SNnu-DM scattering at scattering angle psi for DM
    
    Input
    ------
    Ev: The initial SN neutrino energy, MeV
    Tx: The DM kinetic energy after boost, MeV
    mx: DM mass, MeV
    mV: mediator mass, MeV
    psi: The DM scattering angle after boost, rad
    gV: the DM-nu coupling constant
    gD: the DM-DM coupling constant
    max_Ef: Upper limit for the root finding algortihm to find the solution.
            Try to increase if get_Ef_phi returns error
    
    Output
    ------
    scalar: cm^2
    """
    # Get the associated initial nu energy and its scattering angle phi
    Ev,phi,flag,_ = get_Ef_phi(Tx,mx,psi,max_Ef)
    
    # Calculating the DM-nu cross section at angle psi and converts it into cm^2
    if flag == 'valid':
        crox = (gV*gD)**2*diff_sig_lab(Ev,mx,mV,phi)*to_cm2
    elif flag == 'invalid': # if it is invalid, the inputs cannot happen
        crox = 0
    return crox


def sig_xe(Tx,mx,mV,eps,gD):
    """
    The total cross section for DM-electron scattering in the detector
    
    Input
    ------
    Tx: The DM kinetic energy after boost, MeV
    mx: DM mass, MeV
    mV: mediator mass, MeV
    eps: the kinetic mixing parameter
    gD: the DM-DM coupling constant
    
    Output
    ------
    scalar: cm^2
    """
    Ex = Tx+mx    # The DM total energy
    s = mx**2+me**2+2*Ex*me    # The s channel
    tp,tm = get_tpm(mx,me,s)    # The allowed range for t channel
    e2 = 4*np.pi/137    # e^2 in terms of fine structure constant
    
    # evaluate cross section
    dsdt = lambda t: diff_sig_fi(s,t,mx,me,mV)*to_cm2*(eps*gD)**2*e2
    crox,_ = quad(dsdt,tm,tp)
    return crox