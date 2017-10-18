import numpy as np

def molecular_frac(SFR, T, rho, Habundance, mu=1.22, proton_mass=1.6726219E-24, gamma=4./3., fH=0.752, T0=8.E3):
    SFR = np.copy(SFR)
    P = rho * T / (mu * proton_mass)
    rho0 = 0.1 * proton_mass / fH
    P0 = rho0 * T0 / (mu * proton_mass)
    P_jeans = P0 * np.power(rho / rho0, gamma)
    P_margin = np.log10(P / P_jeans)
    SFR[P_margin > .5] = 0
    return np.where(SFR > 0, 1. / (1. + 1. / np.power(P / 4.3E4, .92)), 0.)

#function below based on Rahmati+ 2013
def neutral_frac(redshift, nH, T, onlyA1=False, noCol=False, onlyCol=False, SSH_Thresh=False, local=False, APOSTLE_corrections=False, SFR=None, mu=1.22, proton_mass=1.6726219E-24, gamma=4./3., fH=0.752, Habundance=None, T0=8.E3, rho=None):

    #APOSTLE pre-treatment for gas temperature
    if APOSTLE_corrections:
        T = np.copy(T)
        SFR = np.copy(SFR)
        P = rho * T / (mu * proton_mass)
        rho0 = 0.1 * proton_mass / fH
        P0 = rho0 * T0 / (mu * proton_mass)
        P_jeans = P0 * np.power(rho / rho0, gamma)
        P_margin = np.log10(P / P_jeans)
        SFR[P_margin > .5] = 0
        T_jeans = mu * Habundance * P_jeans * nH
        T[np.logical_or(SFR > 0, np.logical_and(P_margin < .5, T_jeans > 1.E4))] = 1.E4
        

    # --------------------------------------------------------------------------------------------
    #+
    # NAME:
    #       rahmati2013_neutral_frac
    #
    # PURPOSE:
    #       Computes particle neutral fractions based on the fitting functions of 
    #       Rahmati et al. (2013a). By default, it uses the parameters of Table A1 
    #       (based on small cosmological volumes) for z > 1, and of Table A2 (based 
    #       on a 50Mpc volume) for z < 1, to better account for the effects of 
    #       collisional ionisation on the self-shielding density. 
    #
    # CATEGORY:
    #       I/O, HDF5, EAGLE, HI
    #
    # CALLING SEQUENCE:
    #       NeutralFraction = rahmati2013_neutral_frac(redshift, nH, T)
    #       To compute neutral (HI+H_2) mass of particle, multiply NeutralFraction by 
    #       xH and ParticleMass
    #
    # INPUTS:
    #       redshift:        Redshift of snapshot.
    #       nH:              hydrogen number density of the gas 
    #       T:               temperature of the gas 
    #
    # KEYWORD ARGUMENTS:
    #       onlyA1:          routine will use Table A1 parameters for z < 0.5     
    #       noCol:           the contribution of collisional ionisation to
    #       the overall ionisation rate is neglected
    #       onlyCol:         the contribution of photoionisation to
    #       the overall ionisation rate is neglected
    #       SSH_Thresh:      all particles above this density are assumed
    #       to be fully shielded (f_neutral=1)
    #
    # OUTPUTS:
    #       Array containing neutral fractions
    #
    # DEVELOPMENT:
    #       Written by Rob Crain, Leiden, March 2014, with input from Ali
    #       Rahmati. Based on Rahmati et al. (2013)
    #       Translated from IDL to Python2.7 by Kyle Oman, Victoria, December 2015
    #
    # --------------------------------------------------------------------------------------------

    if (redshift >= 0.0) and (redshift < 1.0):
        dz = redshift
        if onlyA1:
            lg_n0_lo     = -2.94
            gamma_uvb_lo =  8.34e-14
            alpha1_lo    = -3.98
            alpha2_lo    = -1.09
            beta_lo      =  1.29
            f_lo         =  0.01
        else:
            lg_n0_lo     = -2.56
            gamma_uvb_lo =  8.34e-14
            alpha1_lo    = -1.86
            alpha2_lo    = -0.51
            beta_lo      =  2.83
            f_lo         =  0.01
            
        lg_n0_hi     = -2.29
        gamma_uvb_hi =  7.3e-14
        alpha1_hi    = -2.94
        alpha2_hi    = -0.90
        beta_hi      =  1.21
        f_hi         =  0.03
        
    elif (redshift >= 1.0) and (redshift < 2.0):
        dz = redshift - 1.0

        lg_n0_lo     = -2.29
        gamma_uvb_lo =  7.3e-14
        alpha1_lo    = -2.94
        alpha2_lo    = -0.90
        beta_lo      =  1.21
        f_lo         =  0.03     
        
        lg_n0_hi     = -2.06
        gamma_uvb_hi =  1.50e-12
        alpha1_hi    = -2.22
        alpha2_hi    = -1.09
        beta_hi      =  1.75
        f_hi         =  0.03
        
    elif (redshift >= 2.0) and (redshift < 3.0):
        dz = redshift - 2.0

        lg_n0_lo     = -2.06
        gamma_uvb_lo =  1.50e-12
        alpha1_lo    = -2.22
        alpha2_lo    = -1.09
        beta_lo      =  1.75
        f_lo         =  0.03     

        lg_n0_hi     = -2.13
        gamma_uvb_hi =  1.16e-12
        alpha1_hi    = -1.99
        alpha2_hi    = -0.88
        beta_hi      =  1.72
        f_hi         =  0.04   
        
    elif (redshift >= 3.0) and (redshift < 4.0):
        dz = redshift - 3.0

        lg_n0_lo     = -2.13
        gamma_uvb_lo =  1.16e-12
        alpha1_lo    = -1.99
        alpha2_lo    = -0.88
        beta_lo      =  1.72
        f_lo         =  0.04     

        lg_n0_hi     = -2.23
        gamma_uvb_hi =  7.91e-13
        alpha1_hi    = -2.05
        alpha2_hi    = -0.75
        beta_hi      =  1.93
        f_hi         =  0.02  
        
    elif (redshift >= 4.0) and (redshift < 5.0):
        dz = redshift - 4.0

        lg_n0_lo     = -2.23
        gamma_uvb_lo =  7.91e-13
        alpha1_lo    = -2.05
        alpha2_lo    = -0.75
        beta_lo      =  1.93
        f_lo         =  0.02     

        lg_n0_hi     = -2.35
        gamma_uvb_hi =  5.43e-13
        alpha1_hi    = -2.63
        alpha2_hi    = -0.57
        beta_hi      =  1.77
        f_hi         =  0.01 
             
    else:
        print "Invalid redshift > 5.0 or < 0.0"
        raise ValueError
        
    
    lg_n0     = lg_n0_lo     + dz * (lg_n0_hi     - lg_n0_lo)
    n0        = np.power(1.0e1, lg_n0)
    gamma_uvb = gamma_uvb_lo + dz * (gamma_uvb_hi - gamma_uvb_lo)
    alpha1    = alpha1_lo    + dz * (alpha1_hi    - alpha1_lo)
    alpha2    = alpha2_lo    + dz * (alpha2_hi    - alpha2_lo)
    beta      = beta_lo      + dz * (beta_hi      - beta_lo)
    f         = f_lo         + dz * (f_hi         - f_lo)

    gamma_ratio = (1.0e0 - f) * np.power(1.0e0 + np.power(nH / n0, beta), alpha1)
    gamma_ratio = gamma_ratio + f * np.power(1.0e0 + nH / n0, alpha2)
    gamma_phot  = gamma_uvb * gamma_ratio

    if local:
        gamma_local = 1.3e-13 * np.power(nH, 0.2) * np.power(T / 1.0e4, 0.2)
        gamma_phot = gamma_phot + gamma_local

    Lambda    = 315614.0e0 / T
    AlphaA    = 1.269e-13 * np.power(Lambda, 1.503e0)
    AlphaA    = AlphaA / np.power(1.0e0 + np.power(Lambda / 0.522e0, 0.470e0), 1.923e0)
    LambdaT   = 1.17e-10 * np.sqrt(T) * np.exp(-157809.0e0 / T) / (1.0e0 + np.sqrt(T / 1.0e5))

    if noCol:
        LambdaT = 0.0e0
    if onlyCol:
        gamma_phot = 0.0e0
    
    A = AlphaA + LambdaT
    B = 2.0e0 * AlphaA + (gamma_phot / nH) + LambdaT
    sqrt_arg = np.power(B, 2) - 4.0e0 * A * AlphaA
    sqrt_arg[sqrt_arg < 0.0e0] = 0.0e0
    sqrt_term = np.sqrt(sqrt_arg)
    f_neutral = (B - sqrt_term) / (2.0e0 * A)

    if SSH_Thresh:
        f_neutral[nH > SSH_Thresh] = 1.0e0

    return f_neutral
