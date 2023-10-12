# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:04:10 2022

@author: PKollepara
"""

import numpy as np
from scipy import linalg
import scipy.optimize as so

POS_ZERO = 1e-2 #The positive alternative to zero to avoid singularities
MAX_TRIALS = 20

#This function returns the growth rate given a susceptible state vector s, transmission matrix B, and recovery rate g
def growth_rate(s, B, g): 
    tmp = (B.T * s).T 
    M = tmp - np.eye(len(s))*g 
    EV = linalg.eigvals(M)
    gr = np.amax(EV) #Growth rate
    return gr 

# Equation for the final size x, n is the group size, g is recovery rate
def FS(x, B, n, g): 
    return [x[k] - n[k]*(1-np.exp(-np.sum(B[k]*x/g))) for k in range(len(n))]

#Expression for linear combination of final sizes 'x'
def LCFS(a, x): 
    return np.sum(a*x) 

#Function to compute final size
def compute_FS(n, B, g, trials = MAX_TRIALS):    
    for count in range(trials):
        root = so.root(FS, np.random.uniform(0.5, 1)*n, args = (B, n, g)) 
        if root.success == True and np.all(root.x >= np.zeros(len(n))) and np.all(root.x <= n):
            break
    if root.success == False:
        return np.array([-1, -1])
    else:
        return root.x

#Selects the final sizes from the analytical solutions which minimise LCFS
def choose_minimum_cost(n, B, g, A):     
    if len(n) != 2:
        print('Function works only for two groups')
        return 0
    else:
        p, q = A[0], A[1]
        M11, M12, M21, M22 = B[0, 0]/g, B[0, 1]/g, B[1, 0]/g, B[1, 1]/g
        n1, n2 = n[0], n[1]
        D = M11*M22 - M12*M21
        #s vectors
        possible_solutions = [[0, 1/M22], 
        [n1, (1-n1*M11)/(M22 - n1*D)], 
        [1/M11, 0], 
        [(1-n2*M22)/(M11 - n2*D), n2], 
        [(M22 + np.sqrt(M12*M21*q/p))/D, (M11 + np.sqrt(M12*M21*p/q))/D], 
        [(M22 - np.sqrt(M12*M21*q/p))/D, (M11 - np.sqrt(M12*M21*p/q))/D]]
        
        costs = []
    
        for soln in possible_solutions:
            soln = np.array(soln)
            if np.all(soln>=0) and np.all(soln<=n) and growth_rate(soln, B, g)<POS_ZERO:
                costs.append(np.sum(A*soln))
            else:
                costs.append(-np.inf)
        solution = possible_solutions[np.argmax(costs)]
        solution_type = np.argmax(costs)
        mcx = n - solution
    return mcx, solution_type
    

#Equation for change in transmission B_kl --> B_kl (1-c_k)(1-c_l)
def change_transmission(c, B, r, n):
    #return np.array([1 - r[k]/n[k] - np.exp(np.sum(B[k]*(1-c[k])*(1-c)*r)) for k in range(len(n))])
    return np.array([np.sum(B[k]*(1-c[k])*(1-c)*r) + np.log(1-r[k]/n[k]) for k in range(len(n))])

#Computes the vector c. opt_size is the required final size    . Approximates opt size if its too close to group size
# def compute_c(n, B, g, og_size, opt_size, guess, trials = MAX_TRIALS):
#     opt_size_approx = opt_size.copy()
#     for k in range(len(n)):
#         if opt_size_approx[k] > (1-POS_ZERO)*n[k]:
#             opt_size_approx[k] = opt_size_approx[k]*(1-POS_ZERO)
    
#     c_flag = False
#     c = np.array([np.nan, np.nan])
#     min_residue = POS_ZERO
#     for count in range(trials):
#         if np.any(guess != False):
#             root = so.root(change_transmission, guess, args = (B/g, opt_size_approx, n))
#         else:
#             root = so.root(change_transmission, np.random.uniform(-1, 1, len(n)), args = (B/g, opt_size_approx, n))
#         residue = np.sum(np.abs(change_transmission(root.x, B/g, opt_size_approx, n)**2))
#         if residue < min_residue and root.success:
#             c = root.x
#             c_flag = root.success
        
#     if c_flag:    
#         return c
#     else:
#         return np.array([np.nan, np.nan])


#Computes the vector c. opt_size is the required final size. If r[k] >= (1-POS_ZERO)*n[k] we assume that r[k] = n[k] and hard code the analytical result from limit
def compute_c(n, B, g, og_size, opt_size, guess, trials = MAX_TRIALS):
    #opt_size_approx = opt_size.copy()
    
    if np.all(opt_size < (1-POS_ZERO)*n):        
        c_flag = False
        c = np.array([np.nan, np.nan])
        min_residue = POS_ZERO
        for count in range(trials):
            if np.any(guess != False):
                root = so.root(change_transmission, guess, args = (B/g, opt_size, n))
            else:
                root = so.root(change_transmission, np.random.uniform(-1, 1, len(n)), args = (B/g, opt_size, n))
            residue = np.sum(np.abs(change_transmission(root.x, B/g, opt_size, n)**2))
            if residue < min_residue and root.success:
                c = root.x
                c_flag = root.success
    elif opt_size[0] >= (1-POS_ZERO)*n[0]:
        c = np.array([-np.inf, 1])
        c_flag = True
    else:
        c = np.array([1, -np.inf])
        c_flag = True
    
    if c_flag:    
        return c
    else:
        return np.array([np.nan, np.nan])

    
#Computes the vector c. opt_size is the required final size. Return nan if final sizes are too close to group size 
# def compute_c(n, B, g, og_size, opt_size, trials = MAX_TRIALS):
#     if np.any(opt_size >= (1-POS_ZERO)*n):
#         c = np.array([np.nan, np.nan])
#         c_flag = True
#     else:
#         c_flag = False
#         c = np.array([np.nan, np.nan])
#         min_residue = POS_ZERO
#         for count in range(trials):
#             root = so.root(change_transmission, np.random.uniform(-1, 1, len(n)), args = (B/g, opt_size, n))
#             residue = np.sum(np.abs(change_transmission(root.x, B/g, opt_size, n)**2))
#             #c, c_flag = root.x, root.success
#             if residue < min_residue and root.success:
#                 c = root.x#.copy()
#                 c_flag = root.success#.copy()
        
#     if c_flag:    
#         return c
#     else:
#         return np.array([np.inf, np.inf])

#Executes the above defined functions based on the given parameters: n is group size vector, B is transmission matrix and g is recovery rate
def Executor(n, B, g, A, guess, show_output = False):
    gr = growth_rate(n, B, g)
    if gr<POS_ZERO:
        if show_output:
            print('R0 too small (<%f) to compute final size'%POS_ZERO)
    else:
        fs_og = compute_FS(n, B, g)
        fs_opt, sol_type = choose_minimum_cost(n, B, g, A)
        c = compute_c(n, B, g, fs_og, fs_opt.copy(), guess)
        if show_output:
            print('====== Result: Exact Solution ====== \n')
            print('The given cost function weights are: ', A)
            print('For the given group sizes: ', n, ', transmission matrix: \n', B, '\n and recovery rate: ', g)
            print('The basic reproduction number is: ', 1+gr/g)
            print('The final size for the unmitigated epidemic: ', fs_og)
            print('The final size which minimises the cost:     ', fs_opt)
            print('The % change in transmission in the two groups: ', (-c*100).tolist())
            print('The solution type: ', sol_type)
            print('\n====== *** ====== *** ====== *** ====== \n')
        return (fs_og, fs_opt, c, gr, sol_type) 
        
    
    
# n = np.array([0.4, 0.6]) #Define the sizes of sub-populations
# no_g = len(n)
# gamma = 1 #Recovery rate
# #A = np.array([1, 1, 1]) #Weights of the objective function that is minimized. [1, 1, 1] means the final size of the epidemic is minimized

# A = np.array([1, 1])

# b1 = 2
# b2 = 3
# alpha = 0.7 #For parameter variation. B_12 = b1*b2*alpha, B_21 = b1*b2*alpha, M11 = b1^2, B_22 = b2^2

# B = np.array([[b1**2, b1*b2*alpha], [b2*b1*alpha, b2**2]])

# fs_og, fs_opt_analytical, c, gr, sol_type = Executor(n, B, gamma, A, show_output=True)
