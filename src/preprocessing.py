import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR,'data')


def preprocess_pantheonp(path1,z = 0): # The input z here is used to filter redshift values less than z
    data = pd.read_csv(path1, sep = '\s+')
    data_filtered = data[data['zHD'] > z]
    X = data_filtered[['zHD', 'x1', 'c']]
    y = data_filtered['m_b_corr']
    return X,y

def preprocess_des_data(path2, z = 0):
    data = pd.read_csv(path2)
    data_f = data[data['zHD'] > z]
    X = data_f[['zHD','x1','c']]
    y = data_f['mB_corr']
    return X,y