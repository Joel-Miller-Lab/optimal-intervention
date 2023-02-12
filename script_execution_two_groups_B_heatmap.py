# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:53:29 2022

@author: PKollepara
"""

#%% Importing packages and function definitions
import numpy as np
import Optimal_intervention_definition_numerical as numerical
import Optimal_intervention_definition_analytical as exact

POS_ZERO = exact.POS_ZERO  

#%% Selecting type of solution: numerical or exact
smethod = 0
if smethod == 0:
    print('Selected Solution Method: Exact')
else:
    print('Selected Solution Method: Numerical')

#%% Defining system parameters
n = np.array([0.4, 0.6]) #Define the sizes of sub-populations
gamma = 1 

#%% Defining cost function parameters
A = np.array([1, 1])

#%% Define parameter space for the transmission matrix
alpha = 0.7 #For parameter variation. B_12 = b1*b2*alpha, B21 = b1*b2*alpha, B11 = b1^2, B22 = b2^2
b1_list = np.arange(0.05, 3.5, 0.05)
b2_list = np.arange(0.05, 3.5, 0.05)

#%% Empty lists
b1, b2, gr, FS1_int, FS1_ord, FS2_int, FS2_ord, s_type = [], [], [], [], [], [], [], []

#%% Looping over parameters and call Executor function
for index1, x in enumerate(b1_list):
    for index2, y in enumerate(b2_list):
        b1.append(x)
        b2.append(y)
        B = np.array([[x**2, x*y*alpha], [y*x*alpha, y**2]])
            
        tmp = exact.growth_rate(n, B, gamma)
        gr.append(tmp)
        if tmp>POS_ZERO:
            if smethod == 0:                
                fs_og, fs_opt, rate, st = exact.Executor(n, B, gamma, A)
            else:
                fs_og, fs_opt, rate = numerical.Executor(n, B, gamma, A)
            FS1_ord.append(fs_og[0])
            FS2_ord.append(fs_og[1])
            FS1_int.append(fs_opt[0])
            FS2_int.append(fs_opt[1])
            if smethod == 0:
                s_type.append(st)
                
        else:
            FS1_ord.append(float('NaN'))
            FS2_ord.append(float('NaN'))
            FS1_int.append(float('NaN'))
            FS2_int.append(float('NaN'))
            if smethod == 0:
                s_type.append(float('NaN'))

#%% Convert lists in plottable arrays
b1_matrix = np.real(np.reshape(b1, (len(b1_list), len(b2_list))))
b2_matrix = np.real(np.reshape(b2, (len(b1_list), len(b2_list))))
gr_matrix = np.real(np.reshape(gr, (len(b1_list), len(b2_list))))
FS1_ord_matrix = np.real(np.reshape(FS1_ord, (len(b1_list), len(b2_list))))
FS2_ord_matrix = np.real(np.reshape(FS2_ord, (len(b1_list), len(b2_list))))
FS1_int_matrix = np.real(np.reshape(FS1_int, (len(b1_list), len(b2_list))))
FS2_int_matrix = np.real(np.reshape(FS2_int, (len(b1_list), len(b2_list))))
b1, b2 = np.array(b1), np.array(b2)
if smethod == 0:
    s_type_matrix = np.real(np.reshape(s_type, (len(b1_list), len(b2_list))))


#%% Plot settings and importing packages
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap

plt.rcParams['axes.linewidth'] = 0.4
plt.rcParams['hatch.linewidth'] = 0.2

FontSize = 8

plt.rc('font', size=FontSize)          # controls default text sizes
plt.rc('axes', titlesize=FontSize)     # fontsize of the axes title
plt.rc('axes', labelsize=FontSize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FontSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FontSize)    # fontsize of the tick labels
plt.rc('legend', fontsize=FontSize)    # legend fontsize

#%% Suppress plot window
plt.ioff()


#%%Plotting
#%%% Hatch variables for R0<1 region
b1_NA_max = np.amax(np.array(b1)[np.array(gr)<POS_ZERO])
b2_NA_max = np.amax(np.array(b2)[np.array(gr)<POS_ZERO])

#%%% Plotting og final size
plot1 = FS1_int_matrix.T
plot2 = FS2_int_matrix.T

fig, ax = plt.subplots(1, 2, figsize = (5*3.5/8*2.2, 5*3.5/8), constrained_layout = True) 
plt.suptitle(r'$n_1 = %.2f, \ n_2 = %.2f, \ \alpha=%.2f \ G_{kl} = n_k b_k b_l ((1-\alpha)\delta_{kl} + \alpha)$'%(n[0], n[1], alpha) + '\n' + r'Contours show original final size of each group' +
             '\n' + r'$A$ = [%.2f, %.2f]'%(A[0], A[1]), fontsize = FontSize-1)
custom_cmap_1 = cm.get_cmap('Purples').copy()
custom_cmap_2 = cm.get_cmap('Purples').copy()

custom_cmap_1.set_bad(color='none')
custom_cmap_2.set_bad(color='none')

im1 = ax[0].imshow(plot1, cmap = custom_cmap_1, extent = (b1[0], b1[-1], b2[0], b2[-1]), origin = 'lower', interpolation = 'none', aspect = 'equal')
cb1 = fig.colorbar(im1, ax=ax[0], shrink = 1)
cb1.outline.set_visible(False)
cb1.ax.tick_params(width=0.4, labelsize = FontSize-3)
co1 = ax[0].contour(b1_matrix, b2_matrix, FS1_ord_matrix, colors = 'k', linewidths = 0.3)
ax[0].clabel(co1, inline=True, fontsize=FontSize-3)

im2 = ax[1].imshow(plot2, cmap = custom_cmap_2, extent = (b1[0], b1[-1], b2[0], b2[-1]), origin = 'lower', interpolation = 'none', aspect = 'equal')
cb2 = fig.colorbar(im2, ax=ax[1], shrink = 1)
cb2.outline.set_visible(False)
cb2.ax.tick_params(width=0.4, labelsize = FontSize-3)
co2 = ax[1].contour(b1_matrix, b2_matrix, FS2_ord_matrix, colors = 'k', linewidths = 0.3)
ax[1].clabel(co2, inline=True, fontsize=FontSize-3)

#ax[0].fill_between([b1[0], b1_NA_max], y1 = b2[0], y2 = b2_NA_max, hatch = '+++',
#                  color = 'none', edgecolor = 'gray', zorder = 0, lw = 0)
#ax[1].fill_between([b1[0], b1_NA_max], y1 = b2[0], y2 = b2_NA_max, hatch = '+++',
#                  color = 'none', edgecolor = 'gray', zorder = 0, lw = 0)

ax[0].set_title('Final size in group 1 \n from optimal intervention')
ax[1].set_title('Final size in group 2 \n from optimal intervention')


for ax in ax.reshape(-1):
    #ax.set_facecolor('none')
    
    #Set axis labels
    ax.set_xlabel(r'$b_1$')
    ax.set_ylabel(r'$b_2$')
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #Set limits
    ax.set_xlim(0, )
    ax.set_ylim(0, )

    #Set tick numbers
    ax.locator_params('x', nbins = 4)
    ax.locator_params('y', nbins = 4)

    ax.tick_params('both', which = 'both', direction = 'in', labelsize = FontSize-1, width = 0.4)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

if smethod == 0:
    fig.savefig('analytical_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.pdf'%(n[0], n[1], alpha))
    fig.savefig('analytical_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.png'%(n[0], n[1], alpha), dpi=600)

else: 
    fig.savefig('numerical_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.pdf'%(n[0], n[1], alpha))
    fig.savefig('numerical_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.png'%(n[0], n[1], alpha), dpi=600)
    
#%%% Plotting percentage change in final size
plot1 = ((FS1_int_matrix - FS1_ord_matrix)/FS1_ord_matrix).T*100
plot2 = ((FS2_int_matrix - FS2_ord_matrix)/FS2_ord_matrix).T*100

p1max, p2max, p1min, p2min = np.nanmax(plot1), np.nanmax(plot2), np.nanmin(plot1), np.nanmin(plot2)
pmax = np.amax([p1max, p2max])
pmin = np.amax([p1min, p2min])

fig, ax = plt.subplots(1, 2, figsize = (5*3.5/8*2.2, 5*3.5/8), constrained_layout = True) 
plt.suptitle(r'$n_1 = %.2f, \ n_2 = %.2f, \ \alpha = %.2f \ G_{kl} = n_k b_k b_l ((1-\alpha)\delta_{kl} + \alpha)$'%(n[0], n[1], alpha) + '\n' + r'$A$ = [%.2f, %.2f]'%(A[0], A[1]), fontsize = FontSize-1)

custom_cmap_1 = cm.get_cmap('RdGy_r').copy()
custom_cmap_2 = cm.get_cmap('RdGy_r').copy()
custom_cmap_1.set_bad(color='none')
custom_cmap_2.set_bad(color='none')

divnorm1 = colors.TwoSlopeNorm(vmin=pmin, vcenter=0)#, vmax=pmax)
divnorm2 = colors.TwoSlopeNorm(vmin=pmin, vcenter=  0)#, vmax=pmax)

im1 = ax[0].imshow(plot1, cmap = custom_cmap_1, norm = divnorm1, extent = (b1[0], b1[-1], b2[0], b2[-1]), origin = 'lower', interpolation = 'none', aspect = 'equal')
cb1 = fig.colorbar(im1, ax=ax[0], shrink = 1)
cb1.outline.set_visible(False)
cb1.ax.tick_params(width=0.4, labelsize = FontSize-3)

#ax[0].scatter(b1_c1_neg, b2_c1_neg, marker = 'o', c = 'blueviolet', s = 0.5, lw = 0, alpha = 0.1)

im2 = ax[1].imshow(plot2, cmap = custom_cmap_2, norm = divnorm2, extent = (b1[0], b1[-1], b2[0], b2[-1]), origin = 'lower', interpolation = 'none', aspect = 'equal')
cb2 = fig.colorbar(im2, ax=ax[1], shrink = 1)
cb2.outline.set_visible(False)
cb2.ax.tick_params(width=0.4, labelsize = FontSize-3)

#ax[1].scatter(b1_c2_neg, b2_c2_neg, marker = 'o', c = 'blueviolet', s = 0.5, lw = 0, alpha = 0.1)


ax[0].fill_between([b1[0], b1_NA_max], y1 = b2[0], y2 = b2_NA_max, hatch = '++++',
                  color = 'none', edgecolor = 'gray', zorder = 0, lw = 0)
ax[1].fill_between([b1[0], b1_NA_max], y1 = b2[0], y2 = b2_NA_max, hatch = '++++',
                  color = 'none', edgecolor = 'gray', zorder = 0, lw = 0)
#ax[0].scatter(plot_b1_NA, plot_b2_NA, edgecolor = 'k', facecolor = 'none', marker = 's', s = 1, lw = 0.01)

ax[0].set_title(r'$\%$ change in final size'+ '\n in group 1', fontsize = FontSize - 1)
ax[1].set_title(r'$\%$ change in final size'+ '\n in group 2', fontsize = FontSize - 1)


for ax in ax.reshape(-1):
    #ax.set_facecolor('none')
    
    #Set axis labels
    ax.set_xlabel(r'$b_1$')
    ax.set_ylabel(r'$b_2$')
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #Set limits
    ax.set_xlim(0, )
    ax.set_ylim(0, )

    #Set tick numbers
    ax.locator_params('x', nbins = 4)
    ax.locator_params('y', nbins = 4)

    ax.tick_params('both', which = 'both', direction = 'in', labelsize = FontSize-1, width = 0.4)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

if smethod == 0:
    fig.savefig('analytical_pc_change_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.pdf'%(n[0], n[1], alpha))
    fig.savefig('analytical_pc_change_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.png'%(n[0], n[1], alpha), dpi=600)

else: 
    fig.savefig('numerical_pc_change_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.pdf'%(n[0], n[1], alpha))
    fig.savefig('numerical_pc_change_final_size_n1_%.2f_n2_%.2f_alpha_%.2f.png'%(n[0], n[1], alpha), dpi=600)

#%%% Plotting solution type
if smethod == 0:
    #plot = np.array(s_type).copy()
    plot = s_type_matrix.T.copy()
    #clrs = ['darkslateblue', 'olivedrab', 'firebrick', 
    #        'darkmagenta', 'slategray', 'darkslategray']#, 'gainsboro']
    
    clrs = ['#393b79ff', '#637939ff', '#8c6d31ff', '#843c39ff', '#7b4173ff', '#de9ed6ff']
    custom_cmap = ListedColormap(clrs)
    custom_cmap.set_bad('none')
    
    fig, ax = plt.subplots(1, 2, figsize = (5*3.5/8*2.2, 5*3.5/8), constrained_layout = True) 
    plt.suptitle(r'$n_1 = %.2f, \ n_2 = %.2f, \ \alpha = %.2f$'%(n[0], n[1], alpha) + r'$G_{kl} = n_k b_k b_l ((1-\alpha)\delta_{kl} + \alpha)$' + '\n' + r'$A$ = [%.2f, %.2f]'%(A[0], A[1]), fontsize = FontSize-1)
    
    im1 = ax[0].imshow(plot, extent = (b1[0], b1[-1], b2[0], b2[-1]), vmin = -0.49, vmax = 5.49, 
                       cmap = custom_cmap, origin = 'lower', interpolation = 'none', aspect = 'equal')
    cb1 = fig.colorbar(im1, ax=ax[0], shrink = 1)
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(width=0.4, labelsize = FontSize-3)
    
    #ax[0].plot(b1_list, alpha*b1_list, lw = 0.4, c = 'k')
    #ax[0].plot(b1_list, 1/alpha*b1_list, lw = 0.4, c = 'white', ls = '-.')
    
    im2 = ax[1].imshow(plot, extent = (b1[0], b1[-1], b2[0], b2[-1]), vmin = 0, vmax = 5, 
                       cmap = custom_cmap, origin = 'lower', interpolation = 'none', aspect = 'equal')
    cb2 = fig.colorbar(im2, ax=ax[1], shrink = 1)
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(width=0.4, labelsize = FontSize-3)

    # for i in range(0, 6):
    #     plot_b1 = np.array(b1)[plot==i]
    #     plot_b2 = np.array(b2)[plot==i]
    #     ax[0].scatter(plot_b1, plot_b2, c = clrs[i], marker = 's', s = 1, label = str(i))
    #     ax[1].scatter(plot_b1, plot_b2, c = clrs[i], marker = 's', s = 1, label = str(i))
    
    for j in range(2):
        ax[j].set_title('Solution \n Type')
        ax[j].fill_between([b1[0], b1_NA_max], y1 = b2[0], y2 = b2_NA_max, hatch = '++++',
                          color = 'none', edgecolor = 'k', zorder = 0, lw = 0)
        
        #color_of_s_type = [float('NaN')] * len(s_type)
        #for i in range(0, 7):
        #    color_of_s_type[plot == i] = clrs[i]
        #color_of_s_type = [clrs[int(round(i))] for i in plot]
        #color_of_s_type = ['gainsboro' if x==-1 else x for x in color_of_s_type]
        
        
        #scat = ax.scatter(b1, b2, c = color_of_s_type, marker = 's', cmap = 'tab20b', s = 0.25)
        
        #ax[j].legend(loc = 'lower left', ncol = 3, fontsize = FontSize - 3, fancybox = False)
        #ax.set_title(r'Solution type', fontsize = FontSize - 1)
        
        ax[j].locator_params('x', nbins = 4)
        ax[j].locator_params('y', nbins = 4)
            
        #Set axis labels
        ax[j].set_xlabel(r'$b_1$')
        ax[j].set_ylabel(r'$b_2$')
        
        # Hide the right and top spines
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
    
        #Set limits
        ax[j].set_xlim(0, b1_list[-1])
        ax[j].set_ylim(0, b2_list[-1])
    
        ax[j].tick_params('both', which = 'both', direction = 'in', labelsize = FontSize-1, width = 0.4)
    
        # Only show ticks on the left and bottom spines
        ax[j].yaxis.set_ticks_position('left')
        ax[j].xaxis.set_ticks_position('bottom')
        ax[j].set_aspect('equal')

    plt.savefig('analytical_solution_type_n1_%.2f_n2_%.2f_alpha_%.2f.pdf'%(n[0], n[1], alpha))
    plt.savefig('analytical_solution_type_n1_%.2f_n2_%.2f_alpha_%.2f.png'%(n[0], n[1], alpha), dpi=600)

