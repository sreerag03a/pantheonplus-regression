import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR,'data')

# Load data for preprocessing, this is just a basic function
def load_data(path1):
    data = pd.read_csv(path1, delim_whitespace=True) 
    #Here, delim_whitespace is used because of the data being in a .dat file and the headers are separated by whitespace
    return data

def preprocessor_linear(path1,z):
    data= load_data(path1)
    data_filtered = data[data['zHD'] > z]
    X = data_filtered[['zHD', 'x1', 'c']]
    y = data_filtered['m_b_corr']
    return X,y


# panthpath = os.path.join(data_path,'Pantheon+SH0ES.dat')
# X,y = preprocessor_linear(panthpath, 0.5)

# print(len(y))

# plt.plot(X['zHD'],y)
# plt.show()