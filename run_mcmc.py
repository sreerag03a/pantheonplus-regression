import numpy as np
import pandas as pd
from src.components.handling.utils import load_obj
from src.components.handling.logger import logging
from src.mcmc.bayes_mcmc import run_mcmc
import matplotlib.pyplot as plt
from getdist import plots,MCSamples
sn_DES = pd.read_csv('data/DES-data.csv')
# print(sn_DES.columns)
temp_ = sn_DES[['zHD','x1','c']]
z_DES = np.array(sn_DES['zHD'])
m_DES = np.array(sn_DES['mB_corr'])
merr_DES = np.array(sn_DES['mBERR'])

OHD_data = np.loadtxt('data/OHD_NH_77.txt')
z_OHD = OHD_data[:,0]
H_OHD = OHD_data[:,1]
Herr_OHD = OHD_data[:,2]

logging.info('Loading preprocessor')
preprocessor_path = 'outputs/models/preprocessor.pkl'
preprocessor = load_obj(preprocessor_path)
input_array = preprocessor.transform(temp_)

del temp_
logging.info('preprocessing complete')

model = load_obj('outputs/models/advanced_models/Gaussian Process Regressor with Matern kernel.pkl')
logging.info('Making predictions with : Gaussian Process Regressor with Matern kernel')
mB_pred,mB_err_pred = model.predict(input_array,return_std=True)

del model
del preprocessor

'''
Uncomment the below code if you want to save the predictions to a csv files
'''
# sim_data = pd.DataFrame({'zHD' : z_DES, 'm_b' : mB_pred, 'm_b_err' : mB_err_pred})
# print(sim_data)
# logging.info('Saving simulated dataset into outputs/simulated_DES.csv')
# sim_data.to_csv('outputs/simulated_DES.csv')

# del sim_data

real_data = z_DES.copy(),m_DES.copy().copy(),merr_DES.copy(),z_OHD.copy(),H_OHD.copy(),Herr_OHD.copy()
sim_data = z_DES.copy(),mB_pred.copy(),mB_err_pred.copy(),z_OHD.copy(),H_OHD.copy(),Herr_OHD.copy()
if __name__ == '__main__':
    lcdm_samples,wcdm_samples = run_mcmc(real_data,24,2500) # run_mcmc(data,nwalkers,nsteps)
    lcdm_samples1,wcdm_samples1 = run_mcmc(sim_data,24,2500)
    
    labels1 = [r'$H_0$',r'$\Omega_{m0}$',r'$M$']
    labels2 = [r'$H_0$',r'$\Omega_{m0}$', r'$\omega_0$',r'$M$']
    g = plots.get_subplot_plotter(subplot_size=2)

    mcsamples = MCSamples(samples=lcdm_samples,names = labels1, label = r'$\Lambda$CDM - DES')
    mcsamples1 = MCSamples(samples=wcdm_samples,names = labels2, label = r'$\omega$CDM - DES')
    mcsamples2 = MCSamples(samples=lcdm_samples1,names = labels1, label = r'$\Lambda$CDM - Simulated')
    mcsamples3 = MCSamples(samples=wcdm_samples1,names = labels2, label = r'$\omega$CDM - Simulated')

    names = [r'$\Lambda$CDM - DES',r'$\omega$CDM - DES',r'$\Lambda$CDM - Simulated',r'$\omega$CDM - Simulated']

    g.triangle_plot((mcsamples1,mcsamples2,mcsamples3,mcsamples),filled=True)
    samples_total = [lcdm_samples,wcdm_samples,lcdm_samples1,wcdm_samples1]
    for i,samples in enumerate(samples_total):
        best_fit_params1 = np.median(samples, axis=0)
        parameter_uncertainties1 = np.std(samples, axis=0)
        print(f'{names[i]} : {best_fit_params1}')
        print(f'{names[i]} : {parameter_uncertainties1}')
    plt.show()
