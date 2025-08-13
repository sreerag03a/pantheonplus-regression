# import pandas as pd
import numpy as np
#Single feature linear regressor with gradient descent
def linearregressor(x,y,n_iter, learnrate = 0.05):
    n_data = len(y)
    bias_term = np.c_[np.ones((n_data, 1)), x]
    theta = np.random.randn(2,1)
    for i in range(n_iter):
        gradients = (1/n_data) * bias_term.T.dot(bias_term.dot(theta) - y)
        theta = theta - (learnrate*gradients)
    return theta.flatten(), bias_term #We export the bias term so that we can extract the predictions easily from the parameters as bias_term.dot(parameters)

#Linear regression with multiple features (multiple linear regression)
def multi_linear(x,y,n_iter, learnrate = 0.05):
    n_data = len(y)
    
    bias_term = np.c_[np.ones((n_data, 1)), x]
    nlinear = bias_term.shape[1]
    theta = np.random.randn(nlinear,1)
    for i in range(n_iter):
        gradients = (1/n_data) * bias_term.T.dot(bias_term.dot(theta) - y)
        theta = theta - (learnrate*gradients)
    return theta.flatten(),bias_term

#Polynomial regression with gradient descent
def poly_regressor(x,y,n_iter, learnrate = 0.05, degree = 2):
    n_data = len(y)
    x_bias = np.empty((n_data,degree))
    for i in range(degree):
        n = i+1
        x_bias[:,i] = x[:,0]**n
    bias_term = np.concatenate((np.ones((n_data, 1)),x_bias), axis = 1)
    npoly = bias_term.shape[1]
    theta = np.random.randn(npoly,1)
    for i in range(n_iter):
        gradients = (1/n_data) * bias_term.T.dot(bias_term.dot(theta) - y)
        theta = theta - (learnrate*gradients)
    return theta.flatten(),bias_term


