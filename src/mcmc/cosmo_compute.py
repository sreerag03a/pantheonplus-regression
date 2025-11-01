import numpy as np
from scipy.integrate import quad
from src.mcmc.cythonize import cosmofunctions as csm
c = 3e5

def chisq2_LCDM(params,z_sn,m_sn,m_err):
    H0,omega_m,M_abs = params
    integrals = [quad(csm.inversehubbleparameter_LCDM, 0, z, args = (H0, omega_m))[0]*c*(1+z)for z in z_sn]
    m = 25+ 5*np.log10(integrals) + M_abs
    return csm.chisq_gen(m_sn,m,m_err)

def chisq2_wCDM(params,z_sn,m_sn,m_err):
    H0,omega_m,w,M_abs = params
    integrals = [quad(csm.inversehubbleparameter_wCDM, 0, z, args = (H0, omega_m,w))[0]*c*(1+z) for z in z_sn]
    m = 25+ 5*np.log10(integrals) + M_abs
    return csm.chisq_gen(m_sn,m,m_err)

def chisq_LCDM(params,z_sn,m_sn,m_err,z_h,H,Herr):
    H0,omega_m,M_abs = params
    return csm.chisq1_LCDM(H0,omega_m,z_h,H,Herr) + chisq2_LCDM(params,z_sn,m_sn,m_err)

def chisq_wCDM(params,z_sn,m_sn,m_err,z_h,H,Herr):
    H0,omega_m,w,M_abs = params
    return csm.chisq1_wCDM(H0,omega_m,w,z_h,H,Herr) + chisq2_wCDM(params,z_sn,m_sn,m_err)

def log_prior_LCDM(params):
    H0,omega_m,M_abs = params
    if 40 < H0 < 100 and 0 < omega_m < 1 and -25 < M_abs < -17 :
        return 0.0
    return -np.inf

def log_prior_wCDM(params):
    H0,omega_m,w,M_abs = params
    if 40 < H0 < 100 and 0 < omega_m < 1 and -2 < w < 0 and -25 < M_abs < -17:
        return 0.0
    return -np.inf
