# import pandas as pd
import numpy as np

#Simple gradient descent linear regressor

# class Regressor:
#     def __init__(self, x , y, n_iter, learnrate = 0.05):
#         self.learnrate = 0.05
#         self.n_iterations = n_iter



def linearregressor(x,y,n_iter, learnrate = 0.05):
    n_data = len(y)
    bias_term = np.c_[np.ones((n_data, 1)), x]
    theta = np.random.randn(2,1)
    for i in range(n_iter):
        gradients = (1/n_data) * bias_term.T.dot(bias_term.dot(theta) - y)
        theta = theta - (learnrate*gradients)
    return theta.flatten()

def thirdregressor(x,y,n_iter, learnrate = 0.05):
    n_data = len(y)
    x_sq = x**2
    x_cb = x**3
    x_qd = x**4
    x_pt = x**5
    bias_term = np.c_[np.ones((n_data, 1)), x, x_sq, x_cb,x_qd, x_pt]
    theta = np.random.randn(6,1)
    for i in range(n_iter):
        gradients = (1/n_data) * bias_term.T.dot(bias_term.dot(theta) - y)
        theta = theta - (learnrate*gradients)
    return theta.flatten()

# x_data = np.random.rand(100,1)
# y_data = 4*x_data + 12 + (0.25*np.random.randn(100, 1))

# m,c = linearregressor(x_data,y_data,10000)
# print(x_data)