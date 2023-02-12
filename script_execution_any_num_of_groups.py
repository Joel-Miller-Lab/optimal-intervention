# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 19:28:29 2022

@author: PKollepara
"""
import numpy as np
#from scipy import optimize
#from scipy import linalg
#import scipy.optimize as so

import Optimal_intervention_definition_numerical as numerical

  
n = np.array([0.4, 0.6]) #Define the sizes of sub-populations
no_g = len(n)
gamma = 1 #Recovery rate
#A = np.array([1, 1, 1]) #Weights of the objective function that is minimized. [1, 1, 1] means the final size of the epidemic is minimized

A = np.array([1, 1])

b1 = 1.2
b2 = 2.0
alpha = 1.5 #For parameter variation. B_12 = b1*b2*alpha, B21 = b1*b2*alpha, B11 = b1^2, B22 = b2^2

B = np.array([[b1**2, b1*b2*alpha], [b2*b1*alpha, b2**2]])

fs_og, fs_opt_numerical = numerical.Executor(n, B, gamma, A, True)