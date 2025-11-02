import emcee
import numpy as np
from multiprocessing import Pool
import os
from src.mcmc.cosmo_compute import chisq_LCDM, chisq_wCDM, log_prior_LCDM, log_prior_wCDM

os.environ["OMP_NUM_THREADS"] = "1"



def log_likelihood(params,z_sn,m_sn,m_err,z_h,H,Herr,chisquarefunc):
    return -0.5*chisquarefunc(params,z_sn,m_sn,m_err,z_h,H,Herr)

def log_post(params,log_prior,z_sn,m_sn,m_err,z_h,H,Herr,chisquarefunc):
    prior = log_prior(params)
    if not np.isfinite(prior):
        return -np.inf
    return prior + log_likelihood(params,z_sn,m_sn,m_err,z_h,H,Herr,chisquarefunc)


def run_mcmc(data,n_walkers,n_steps):
    z_sn,m_sn,m_err,z_h,H,Herr = data
    initial_lcdm = [67.0,0.2,-19.0] + 1e-1*np.random.rand(n_walkers,3)
    initial_wcdm = [67.0,0.2,-19.0,-0.7] + 1e-1*np.random.rand(n_walkers,4)
    with Pool() as pool:
        sampler_lcdm = emcee.EnsembleSampler(n_walkers,3,log_post,args = (log_prior_LCDM,z_sn,m_sn,m_err,z_h,H,Herr,chisq_LCDM),pool=pool)
        sampler_lcdm.run_mcmc(initial_lcdm,n_steps,progress=True)
        samples_lcdm = sampler_lcdm.get_chain(discard=1200, thin=10,flat = True)


    with Pool() as pool:
        sampler_wcdm = emcee.EnsembleSampler(n_walkers,4,log_post,args = (log_prior_wCDM,z_sn,m_sn,m_err,z_h,H,Herr,chisq_wCDM),pool=pool)
        sampler_wcdm.run_mcmc(initial_wcdm,n_steps,progress=True)
        samples_wcdm = sampler_wcdm.get_chain(discard=1200, thin=10,flat = True)
    
    return samples_lcdm,samples_wcdm