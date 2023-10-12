# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 22:05:50 2022

@author: PKollepara
"""

import numpy as np
from scipy import optimize
from scipy import linalg
import scipy.optimize as so

POS_ZERO = 1e-3 #The positive alternative to zero to avoid singularities
MAX_TRIALS = 20

#This function returns the growth rate given a susceptible state vector s, transmission matrix B, and recovery rate g
def growth_rate(s, B, g): 
    tmp = (B.T * s).T 
    M = tmp - np.eye(len(s))*g 
    EV = linalg.eigvals(M)
    gr = np.amax(EV) #Growth rate
    return gr 

# Equation for the final size x, n is the group size, g is recovery rate
def FS_eqn(x, B, n, g): 
    return [x[k] - n[k]*(1-np.exp(-np.sum(B[k]*x/g))) for k in range(len(n))]

# Equation for the final size x, n is the group size, g is recovery rate
def FS_fn(x, B, n, g): 
    return np.array([n[k]*(1-np.exp(-np.sum(B[k]*x/g))) for k in range(len(n))])


#Expression for linear combination of final sizes 'x'
def LCFS(a, x): 
    return np.sum(a*x) 

#Function to compute final size
def compute_FS(n, B, g, trials = MAX_TRIALS):    
    for count in range(trials):
        root = so.root(FS_eqn, np.random.uniform(0.5, 1)*n, args = (B, n, g)) 
        if root.success == True and np.all(root.x >= np.zeros(len(n))) and np.all(root.x <= n):
            break
    if root.success == False:
        return np.array([-1, -1])
    else:
        return root.x
    
def compute_FS_alt(n, B, g, max_trials = 50):    
    x = np.random.uniform(0, 1)*n
    i = 0
    while True:
        x_old = x.copy()
        x = FS_fn(x_old, B, n, g)
        i=i+1
        #if np.linalg.norm((x-x_old)/(x_old)) < reltol or 
        if i > max_trials:
            break
    if np.all(x >= 0) and np.all(x <= n):
        return x
    else:
        return -1*np.ones(len(n))
    
#Finds the final sizes which minimise LCFS
def minimise_cost(n, B, g, A, trials = MAX_TRIALS):     
    constraint_growth_rate = optimize.NonlinearConstraint(lambda x: growth_rate(n-x, B, g), lb=0, ub=0) #since x is final size, we pass the s array = n-x to the growth rate function
    min_cost = LCFS(A, n)
    for j in range(trials):
        optimize_ic = np.random.rand(len(n))*n        
        min_cost_sol = optimize.minimize(LCFS, x0 = optimize_ic, args = (A), constraints = (constraint_growth_rate), bounds = [(0, n[k]) for k in range(len(n))])
        if LCFS(A, min_cost_sol.x) < min_cost:
            min_cost = LCFS(A, min_cost_sol.x)
            mcx = min_cost_sol.x
    return mcx

def minimise_cost_alt(n, B, g, A, trials = MAX_TRIALS):     
    constraint_growth_rate = optimize.NonlinearConstraint(lambda x: growth_rate(n-x, B, g), lb=0, ub=0) #since x is final size, we pass the s array = n-x to the growth rate function
    min_cost = LCFS(A, n)
    for j in range(trials):
        optimize_ic = np.random.rand(len(n))*n        
        min_cost_sol = optimize.minimize(LCFS, method = 'trust-ncg', x0 = optimize_ic, args = (A), constraints = (constraint_growth_rate), bounds = [(0, n[k]) for k in range(len(n))])
        if LCFS(A, min_cost_sol.x) < min_cost:
            min_cost = LCFS(A, min_cost_sol.x)
            mcx = min_cost_sol.x
    return mcx


#Equation for change in transmission B_kl --> B_kl (1-c_k)(1-c_l)
def change_transmission(c, B, r, n):
    #return np.array([np.sum(b[0]*(1-c[0])*(1-c)*r) + np.log(1-r[0]/n[0]), np.sum(b[1]*(1-c[1])*(1-c)*r) + np.log(1-r[1]/n[1]), np.sum(b[2]*(1-c[2])*(1-c)*r) + np.log(1-r[2]/n[2])])
    return np.array([np.sum(B[k]*(1-c[k])*(1-c)*r) + np.log(1-r[k]/n[k]) for k in range(len(n))])

#Computes the vector c. opt_size is the required final size    
def compute_c(n, B, g, opt_size, trials = MAX_TRIALS):
    opt_size_approx = opt_size.copy()
    if opt_size[0] == n[0]: 
        opt_size_approx[0] = (1-POS_ZERO)*opt_size_approx[0]
    if opt_size[1] == n[1]:
        opt_size_approx[1] = (1-POS_ZERO)*opt_size_approx[1]

    trial = 0
    c_flag = False
    while c_flag == False:
        c, c_flag = change_transmission(opt_size_approx, B/g, n)
        trial = trial + 1
        if trial == trials:
            break
    return c, c_flag 

def FS_second_wave(x, n, r, B, g):    
    return [np.log((n[k]-x[k])/(n[k]-r[k])) + np.sum(B[k]*(x-r)/g) for k in range(len(n))]
#return np.array([np.log((n[0]-x[0])/(n[0]-r_init[0])) + B[0, 0]/g*(x[0]-r_init[0]) + B[0, 1]/g*(x[1]-r_init[1]), 
    #np.log((n[1]-x[1])/(n[1]-r_init[1])) + B[1, 0]/g*(x[0]-r_init[0]) + B[1, 1]/g*(x[1]-r_init[1])])

    #return [x[k] - n[k]*(1-np.exp(-np.sum(B[k]*x/g))) for k in range(len(n))]
    
def compute_FS_second_wave(n, r, B, g, trials = MAX_TRIALS):
    for count in range(trials):
        root = so.root(FS_second_wave, np.random.uniform(0.75, 1)*n, args = (n, r, B, g)) 
        if root.success == True and np.all(root.x >= np.zeros(len(n))) and np.all(root.x <= n):
            break
    if root.success == False:
        return np.array([-1, -1])
    else:
        return root.x

#Executes the above defined functions based on the given parameters: n is group size vector, B is transmission matrix and g is recovery rate
def Executor(n, B, g, A, show_output = False):
    gr = growth_rate(n, B, g)
    if gr<POS_ZERO:
        if show_output:
            print('R0 too small (<%f) to compute final size'%POS_ZERO)
    else:
        fs_og = compute_FS(n, B, g)
        fs_opt = minimise_cost(n, B, g, A)
        #c = compute_c(n, B, g, fs_opt)
        if show_output:
            print('====== Result: Numerical Solution ====== \n')
            print('The given cost function weights are: ', A)
            print('For the given group sizes: ', n, ', transmission matrix: \n', B, '\n and recovery rate: ', g)
            print('The basic reproduction number is: ', 1+gr/g)
            print('The final size for the unmitigated epidemic: ', fs_og)
            print('The final size which minimises the cost:     ', fs_opt)
            print('The effective R at the obtained optimum :     ', 1 + growth_rate(n - fs_opt, B, g)/g)
            print('\n====== *** ====== *** ====== *** ====== \n')
        return (fs_og, fs_opt, gr) 
        
def Executor_alt(n, B, g, A, show_output = False):
    gr = growth_rate(n, B, g)
    if gr<POS_ZERO:
        if show_output:
            print('R0 too small (<%f) to compute final size'%POS_ZERO)
    else:
        fs_og = compute_FS(n, B, g)
        fs_opt = minimise_cost_alt(n, B, g, A)
        #c = compute_c(n, B, g, fs_opt)
        if show_output:
            print('====== Result: Numerical Solution ====== \n')
            print('The given cost function weights are: ', A)
            print('For the given group sizes: ', n, ', transmission matrix: \n', B, '\n and recovery rate: ', g)
            print('The basic reproduction number is: ', 1+gr/g)
            print('The final size for the unmitigated epidemic: ', fs_og)
            print('The final size which minimises the cost:     ', fs_opt)
            print('\n====== *** ====== *** ====== *** ====== \n')
        return (fs_og, fs_opt, gr) 
    
# =============================================================================
#     
# n = np.array([0.4, 0.25, 0.35]) #Define the sizes of sub-populations
# no_g = len(n)
# gamma = 1 #Recovery rate
# #A = np.array([1, 1, 1]) #Weights of the objective function that is minimized. [1, 1, 1] means the final size of the epidemic is minimized
# 
# A = np.array([1, 1, 1])
# 
# b1 = 1.2
# b2 = 2.0
# alpha = 1.5 #For parameter variation. B_12 = b1*b2*alpha, B21 = b1*b2*alpha, B11 = b1^2, B22 = b2^2
# 
# B = np.array([[b1**2, b1*b2*alpha], [b2*b1*alpha, b2**2]])
# 
# B = np.array([[3, 2, 1.2], [0.8, 3, 1.5], [1.3, 0.9, 1.5]])
# fs_og, fs_opt_numerical = Executor(n, B, gamma, A)
# =============================================================================
