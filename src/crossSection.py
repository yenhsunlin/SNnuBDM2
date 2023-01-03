import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from constants import *
import warnings


# %% User-derined warning
class OutOfBoundWarning(UserWarning):
    pass


# %% Scatering amplitude squared for vector interaction
def scatteringAmplitudeSquared(s,t,m1,m2,mV):
    """
    The spin-averaged amplitude squared for vector interaction
    
    Input
    ------
    s: Mandelstam variables s
    t: Mandelstam variables t
    m1: incoming mass
    m2: mass of the particle to be scattered off, at rest in the beginning 
    mV: mediator mass
    
    Output
    ------
    scalar: spin-averaged amplitude squared, dimensionless
    """
    # Define sum over mass squared and u channel
    massSum = m1**2 + m2**2
    u = 2*massSum - s - t
    
    # Spin-averaged amplitude-squared
    ampSquared = 2/(t - mV**2)**2*(s**2+u**2+4*t*massSum-2*massSum**2)
    return ampSquared


# %% BDM momentum
def get_BDMp(Tx,mx):
    """
    Get the BDM momentum, please do check if the inputs meet the
    physical requirements
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: DM mass
    
    Output
    ------
    px: the BDM momentum
    """
    return np.sqrt(Tx*(Tx + 2*mx))


# %% BDM velocity
def get_BDMv(Tx,mx):
    """
    The BDM velocity in the unit of light speed
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    
    Output
    ------
    vx: VDM velocity
    """
    return np.sqrt(Tx*(Tx+2*mx))/(Tx+mx)


# %% Get the cosine value of the neutrino scattering angle phi
def get_cosPhi(Ev,Tx,mx,cosPsi):
    """
    Get the cos(phi) where phi is the neutrino scattering angle,
    please do check if the inputs meet the physical requirements
    
    Input
    ------
    Ev: the initial neutrino energy
    Tx: the BDM kinetic energy
    mx: the DM mass
    cosPsi: the cosine value of DM scattering angle, cos(psi)
    
    Output
    ------
    cos(phi): the cosine value of the neutrino scattering phi
    """
    if cosPsi > 1:
        cosPsi = 1
    else: pass
    sinPsiSquared = 1 - cosPsi**2
    return 1/np.sqrt(1 + sinPsiSquared/(Ev/get_BDMp(Tx,mx) - cosPsi)**2)


# %% Get the neutrino energy after scattering
def get_Ev_prime(Ev,Tx,mx,cosPsi):
    """
    Neutrino energy after scattering, Ev_prime, please do check if the
    inputs meet the physical requirements
    """
    return Ev*mx/(mx + Ev*(1 - get_cosPhi(Ev,Tx,mx,cosPsi)))


# %% Get the required Ev and cos(phi) for BDM with kinetic energy Tx at scattering
# angle psi
def get_Ev_cosPhi(Tx,mx,cosPsi,max_Ev = 1000):
    """
    Get the initial neutrino energy and scattering angle cos(phi) for
    a given (Tx,mx,psi). Additional \'flag\' and \'msg\' will be output
    to indicate the set of solution is valid or not
    
    Input
    ------
    Tx: BDM kinetic energy
    mx: DM mass
    cosPsi: the cosine value of DM scattering angle, cos(psi)
    max_Ev: Maximum Ev to be searched for the solution of the algorithm
    
    Output
    ------
    tuple
    Ev: Required initial neutrino energy
    cos(phi): the cosine value of the neutrino scattering
    flag: \'valid\' or \'invalid\' for the solution
    msg: 1: Pass, if flag returns \'valid\'
         2: Solution exists but might not be physical because energy-
            momentum conservation is violated. Such violation could be
            due to round-off error. Further check is necessary
         3: The algorithm cannot find the solution for Ev for the given
            \'max_Ev\'.
    """
    # Equation for getting Tx
    def _Tx(Ev):
        return Ev - get_Ev_prime(Ev,Tx,mx,cosPsi)
    # Target function for root_scalar to find Ev -> Ev_prime - Ev = Tx
    def _f(Ev):
        return _Tx(Ev) - Tx
    # Try to solve the Ev with root_scalar
    try:
        Ev = root_scalar(_f, bracket=[0, max_Ev], method='brentq').root
        # Get phi via arccos instead of arctan to aviod minus phi 
        cosPhi = get_cosPhi(Ev,Tx,mx,cosPsi)
        # Check the energy-momentum conservation
        if np.sqrt(1 - cosPhi**2)/cosPhi >= 0:
            # pass!
            flag = 'valid'
            msg = 1
        else:
            # solution exists but energy-momentum conservation is violated
            flag = 'invalid'
            msg = 2
    except:
        # root_scalar cannot find Ev in the given range
        Ev = np.nan
        cosPhi = np.nan
        flag = 'invalid'
        msg = 3

    return Ev,cosPhi,flag,msg


# %% Differential Nu-DM scattering cross section
"""
def diffCrossSectionNuDM(psi,Tx,mx,mV,gV,gD,max_Ev = 2000):
    
    # following is the docstring
    Lab-frame differential cross section for neutrino-DM scattering
    
    Input
    ------
    psi: the DM scattering angle in rad
    Tx: the BDM kinetic energy in MeV
    mx: DM mass in MeV
    mV: mediator mass in MeV
    gV: the nu-DM coupling constant
    gD: the DM-DM coupling constant
    max_Ev: maximum Ev to be searched for the solution of the algorithm
    
    Output
    ------
    nu-DM diff. cross section: Lab-frame with unit 1/cm^2 * 1/rad
    # above is the docstring
    
    # Get the corresponding Ev and cos(phi)
    Ev,cosPhi,flag,msg = get_Ev_cosPhi(Tx,mx,psi,max_Ev = max_Ev)
    
    # Check if the inputs are physical
    if msg == 1:
        # The inputs resulted physical consequence
        
        # Get the neutrino energy after scattering
        Ev_prime = get_Ev_prime(Ev,Tx,mx,psi)
        
        # The Mandelstam variables
        s = 2*Ev*mx + mx**2
        t = -2*Ev*Ev_prime*(1 - cosPhi)
        
        # The scattering amplitude squared
        amp = scatteringAmplitudeSquared(s,t,0,mx,mV)
        
        # The lab-frame differential cross section
        diffCrox = (1/32/np.pi)/(mx + Ev*(1 - cosPhi))**2*amp*(gV*gD)**2*to_cm2
        return diffCrox
    elif msg == 2:
        # Enery-momentum conservation is violated, the corresponding cross section should be zero
        return 0
    else:
        # Ev cannot be found due to the solution might be outside the max_Ev
        # Try to increase it
        warnings.warn(f'Ev might be outsdie max_Ev = {max_Ev}, try to increase the value and do again',OutOfBoundWarning)
        return 0
"""
    

def diffCrossSectionNuDM(cosPhi,Ev,mx,mV,gV,gD):
    """
    Lab-frame differential cross section for neutrino-DM scattering
    
    Input
    ------
    cosPhi: the cos(phi) where phi is the neutrino scattering angle
    Ev: the initial neutrino energy
    mx: DM mass in MeV
    mV: mediator mass in MeV
    gV: the nu-DM coupling constant
    gD: the DM-DM coupling constant
    
    Output
    ------
    nu-DM diff. cross section: Lab-frame with unit 1/cm^2 * 1/rad
    """ 
    # Get the neutrino energy after scattering
    Ev_prime = Ev*mx/(mx + Ev*(1 - cosPhi))
        
    # The Mandelstam variables
    s = 2*Ev*mx + mx**2
    t = -2*Ev*Ev_prime*(1 - cosPhi)
        
    # The scattering amplitude squared
    amp = scatteringAmplitudeSquared(s,t,0,mx,mV)
        
    # The lab-frame differential cross section
    diffCrox = (1/32/np.pi)/(mx + Ev*(1 - cosPhi))**2*amp*(gV*gD)**2*to_cm2
    return diffCrox


# %% Frame-independent differential DM-e scattering cross section
def diffCrossSectionElectronDM(s,t,mx,mV):
    """
    Frame-independent differential cross section for DM-electron scattering
    
    Input
    ------
    s: Mandelstam variables s
    t: Mandelstam variables t
    mx: the DM mass in MeV
    mV: mediator mass in MeV
    
    Output
    ------
    DM-e diff. cross section: Frame-independent differential cross section  with unit 1/MeV^3
    """
    pStarSquared = (s - (me + mx)**2)*(s - (me - mx)**2)/4
    # the amplitude squared
    amp = scatteringAmplitudeSquared(s,t,mx,me,mV)
    diffCrox = amp/(64*np.pi*pStarSquared)
    return diffCrox


# %% Kallen-lambda function
def KallenLambda(x,y,z):
    """
    The Kallen lambda function
    """
    return x**2 + y**2 + z**2 - 2*(x*y + y*z + z*x)


# %% Get the integration range for evaluating frame-independent cross section
def get_tpm(m1,m2,s):
    """
    Get the integration range for frame-independent cross section
    
    Input
    ------
    m1: the particle that is going to scatter
    m2: the particle that is going to be scattered off
    s: the s-channel
    
    Output
    ------
    tuple: the allowd range for t: (t_max,t_min)
    """
    E_star2 = (s + m1**2 - m2**2)**2/4/s
    p_star2 = KallenLambda(s,m1**2,m2**2)/4/s
    tp = 2*m1**2 -2*E_star2 + 2*p_star2
    tm = 2*m1**2 -2*E_star2 - 2*p_star2
    return tp,tm


# %% Total DM-e scattering cross section
def totalCrossSectionElectronDM(Tx,mx,mV,eps,gD):
    """
    The total cross section for DM-electron scattering in the detector
    
    Input
    ------
    Tx: The DM kinetic energy after boost, MeV
    mx: DM mass, MeV
    mV: mediator mass, MeV
    B: [bV,bA], coefficients for vector and axial-vector interactions at vertex
        Gamma. B = [1,0] indicates vector interaction only
    C: [cV,cA], coefficients for vector and axial-vector interactions at vertex
        Gamma_prime. C = [1,0] indicates vector interaction only
    
    Output
    ------
    DM-e cross section: cm^2
    """
    # DM total energy
    Ex = Tx + mx  
    s = mx**2 + me**2 + 2*Ex*me
    # Integration range for t channel
    tp,tm = get_tpm(mx,me,s)
    # e^2 in terms of the fine structure constant
    e2 = 4*np.pi/137
    
    # Evaluating cross section
    dsdt = lambda t: diffCrossSectionElectronDM(s,t,mx,mV)*to_cm2*(eps*gD)**2*e2
    totCrox,_ = quad(dsdt,tm,tp)
    return totCrox