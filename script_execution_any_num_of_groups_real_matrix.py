# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 19:28:29 2022

@author: PKollepara
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Optimal_intervention_definition_numerical as numerical
import time
import io
import base64

#%%
plt.style.use('seaborn-v0_8-colorblind')
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 8
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rc('axes', titlesize=plt.rcParams['font.size'])     # fontsize of the axes title

#%% html plot saver function
def fig_html(f, html):
    img = io.BytesIO()
    f.savefig(img, format='png', dpi = 400)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    html.write('<img src="data:image/png;base64,{}" alt="Matplotlib Plot" class="blog-image">'.format(plot_url))


#%%
# =============================================================================
#  Wallinga 2006 10.1093/aje/kwj317 
#  Title: Using Data on Social Contacts to Estimate Age-specific Transmission Parameters for Respiratory-spread Infectious Agents
#  Appendix Table 1
#  Age groups: Dutch population (*1000)
#      0  : 184
#      1-5: 876
#      6-12: 1265
#      13-19: 1642
#      20-39: 4857
#      40-59: 3312
#      60-  : 2477
#      Total: 14614
#  Using only ages 1 and above     
# =============================================================================
pop_dict = {'1-5': 876, '6-12': 1265, '13-19': 1642, '20-39': 4857, '40-59': 3312, '60-': 2477}
n = np.array(list(pop_dict.values()))/np.sum(list(pop_dict.values())) #Define the sizes of sub-populations
no_g = len(n)
df_cm = pd.read_csv('Wallinga_2006_contact_matrix.csv', index_col='Age class')
M = df_cm.to_numpy(copy = True)

#%%
gamma = 1 #Recovery rate
#A = np.ones(no_g) # Objective function
#A = np.array([3, 1/2, 1, 2.5, 1.5, 3]) #USA 1918


#CFR, suptitle = np.array([1, 1, 1, 1, 1, 4]), 'Test1'
#CFR, title = np.array([1, 0.76, 2.12, 18.7, 127.86, 283.47]), 'COVID-19, Lancet, 2022'
#CFR, suptitle = np.array([.96, 0.38, .4, 2.6, 5.4, 5.9])/100, 'H1N1, Nishiura 2010'
#CFR, suptitle = np.array([1.68, 0.47, .82, 2.73, 1.49, 3.78])/100, '1918 Pandemic, Taubenberger 2006'
CFR, suptitle = np.array([0.0031, 0.000815, 0.0028, 0.02135, 0.18075, 2.952])/100, 'COVID-19 Pandemic, Driscoll 2021'
#CFR, suptitle = np.array([1, 1, 1, 1, 1, 1]), 'Neutral cost function'
#CFR, suptitle = np.array([3, 3, 3, 1, 1, 1]), 'Younger groups prioritised'

A = CFR/np.sum(CFR)
#A = CFR/np.min(CFR)

q_list = np.arange(0.01, 0.13, 0.001)
#q_list = [0.16]
res_list = []
for q in q_list:    
    keys = [(key1, key2) for key1 in pop_dict.keys() for key2 in pop_dict.keys()]
    
    G = q*M
    B = np.zeros(G.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[0]):
            B[i, j] = G[i, j]/n[i]
    
    try: 
        fs_og, fs_opt_numerical, gr = numerical.Executor(n, B, gamma, A, False)
        R_eff = 1 + numerical.growth_rate(n-fs_opt_numerical, B, gamma)
        
        #res_list.append((f'q = {q} \nR_0 = {1+gr/gamma} \nOriginal final size (ratio w group size): \n{fs_og/n}, Optimal final size (ratio w group size): \n{fs_opt_numerical/n}'))
        res = {'q': q, 'R_0': 1+gr/gamma, 'R_eff': R_eff, 'fs_og': fs_og, 'fs_opt': fs_opt_numerical}
        if np.real(R_eff) > 1+1e-3: ##This used to be <1e-3 which was wrong and was yielding incorrect solutions
            print(f'Warning, q = {q}, R_eff = {R_eff}')
        res_list.append(res)
    except:
        continue
        
#%%
ages = df_cm.columns.values
ages_wts = []
for i, age in enumerate(ages):
    ages_wts.append(f'{age} ({A[i]:.4f})')


for res in res_list:
    X = np.sum(res['fs_opt'])
    res['fs_sum_opt'] = X
df_fs_sum_opt = pd.DataFrame(res_list)[['q', 'R_0', 'R_eff', 'fs_sum_opt']]

for res in res_list:
    X = (res['fs_opt'] - res['fs_og'])/res['fs_og']*100
    for age, x in zip(ages_wts, X):
        res[age] = x
df_ar_pc = pd.DataFrame(res_list)
        
for res in res_list:
    X = (res['fs_opt'] - res['fs_og'])
    for age, x in zip(ages_wts, X):
        res[age] = x
df_ar_diff = pd.DataFrame(res_list)

for res in res_list:
    X = (res['fs_og'])
    for age, x in zip(ages_wts, X):
        res[age] = x
df_ar = pd.DataFrame(res_list)

for res in res_list:
    X = (res['fs_opt'] - res['fs_og'])*CFR/(res['fs_og']*CFR)*100
    for age, x in zip(ages_wts, X):
        res[age] = x
df_fr_pc = pd.DataFrame(res_list)
        
for res in res_list:
    X = (res['fs_opt'] - res['fs_og'])*CFR
    for age, x in zip(ages_wts, X):
        res[age] = x
df_fr_diff = pd.DataFrame(res_list)

for res in res_list:
    X = (res['fs_og'])*CFR
    for age, x in zip(ages_wts, X):
        res[age] = x
df_fr = pd.DataFrame(res_list)


for res in res_list:
    x = res['fs_opt'] - res['fs_og']
    infections_prevented = -(x[x<0]).sum()
    infections_caused = (x[x>0]).sum()
    res['severity_infections'] = infections_caused/infections_prevented
    
    x = (res['fs_opt'] - res['fs_og'])*CFR
    deaths_prevented = -(x[x<0]).sum()
    deaths_caused = (x[x>0]).sum()
    res['severity_deaths'] = deaths_caused/deaths_prevented
df_si = pd.DataFrame(res_list, columns = ['R_0', 'severity_infections'])
df_sd = pd.DataFrame(res_list, columns = ['R_0', 'severity_deaths'])


#df.drop(labels = ['fs_opt', 'fs_og'], inplace = True)

#df.to_html('Wallinga_2006_optimal.html')
#df.to_csv('Wallinga_2006_optimal.csv')

#df[df.flag == 'trolley'].to_html('view_trolley.html')
#f, axs = plt.subplots(1, 1, figsize = (2.5, 3), constrained_layout = True)

#%%
f, axs = plt.subplots(2, 4, figsize = (9.5, 6.5), constrained_layout = True)
num_ticks_hm = 10
#plt.suptitle(suptitle)
titles = [ 'Epidemic size (no intervention)', 'Relative change in epidemic size (%)', 'Change in epidemic size', 'severity']
labels = ['1A', '1B', '1C', '1D']
for df, ax, title, label in zip([df_ar, df_ar_pc, df_ar_diff, df_si], axs[0], titles, labels):
    df.R_0 = np.real(df.R_0).round(2)
    R_0_list = df.R_0.values
    #df = df[df.R_0 < 6]
    df = df.set_index('R_0')
    
    # df = df[ages_wts]
    # if title == 'severity':
    #     prevented = -df[df<0].sum(axis = 1)
    #     caused = df[df>0].sum(axis = 1)
    #     df['severity'] = caused/prevented
    #     ax.plot(df.index.values, df.severity.values, c = 'darkslateblue', lw = 0.25, marker = 'o', ms = 0.4)
    #     ax.set(ylim = (0, df.severity.max()), title = 'Severity of dilemma (infections)')
    # else:
    #     sns.heatmap(df, cmap = 'RdBu_r', center = 0, ax = ax, yticklabels=12)
    #     ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    #     ax.vlines([1, 2, 3, 4, 5], ax.get_ylim()[0], ax.get_ylim()[1], color = 'white')
    #     ax.set_title(title)
    #     ax.set_ylabel('Basic reproduction number')
    # ax.text(-0.15, 1.15, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')    
    if title == 'severity':
        ax.plot(df.index, df.severity_infections, c = 'darkslateblue', lw = 0.5, marker = 's', ms = 0.4)
        ax.set(ylim = (0, df.severity_infections.max()), title = 'Severity of dilemma (infections)')
        
    else:
        df = df[ages_wts]
        # sns.heatmap(df, cmap = 'RdBu_r', center = 0, ax = ax, yticklabels=12)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
        # ax.vlines([1, 2, 3, 4, 5], ax.get_ylim()[0], ax.get_ylim()[1], color = 'white')
        # ax.set_title(title)
        # ax.set_ylabel('Basic reproduction number')
        sns.lineplot(df, ax = ax)
        ax.set_title(title)
        ax.legend(ncols = 1, fontsize = 'small', fancybox = False, )
        if label != '1A':
            ax.hlines(y=0, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1], 
                      colors = 'grey', lw = 0.5)
    ax.set_xlabel(r'$\mathcal{R}_0$')
    ax.text(-0.15, 1.1, f'[{label}]', transform=ax.transAxes, fontsize=10, va='top')    
    ax.grid(True, which = 'both', lw = 0.5, ls = ':')
    ax.set_xlim(1, R_0_list[-1])
        
titles = [ 'Deaths (no intervention)', 'Relative change in deaths (%)', 'Change in deaths', 'severity']
labels = ['2A', '2B', '2C', '2D']
for df, ax, title, label in zip([df_fr, df_fr_pc, df_fr_diff, df_sd], axs[1], titles, labels):
    df.R_0 = np.real(df.R_0).round(2)
    R_0_list = df.R_0.values
    #df = df[df.R_0 < 6]
    df = df.set_index('R_0')
    
    # df = df[ages_wts]
    # if title == 'severity':
    #     prevented = -df[df<0].sum(axis = 1)
    #     caused = df[df>0].sum(axis = 1)
    #     df['severity'] = caused/prevented
    #     ax.plot(df.index.values, df.severity.values, c = 'darkslateblue', lw = 0.25, marker = 'o', ms = 0.4)
    #     ax.set(ylim = (0, df.severity.max()), xlabel = 'Basic reproduction number',
    #            title = 'Severity of dilemma (deaths)')
    # else:
    #     sns.heatmap(df, cmap = 'RdBu_r', center = 0, ax = ax, yticklabels=12)
    #     ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
    #     ax.vlines([1, 2, 3, 4, 5], ax.get_ylim()[0], ax.get_ylim()[1], color = 'white')
    #     ax.set_title(title, fontsize = plt.rcParams['font.size'])
    #     ax.set_ylabel('Basic reproduction number')
    #     ax.set_xlabel('Age group (weight)')
    # ax.text(-0.15, 1.15, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top')    
    if title == 'severity':
        ax.plot(df.index, df.severity_deaths, c = 'darkslateblue', lw = 0.5, marker = 's', ms = 0.4)
        ax.set(ylim = (0, df.severity_deaths.max()), title = 'Severity of dilemma (deaths)')
        
    else:
        df = df[ages_wts]
        # sns.heatmap(df, cmap = 'RdBu_r', center = 0, ax = ax, yticklabels=12)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
        # ax.vlines([1, 2, 3, 4, 5], ax.get_ylim()[0], ax.get_ylim()[1], color = 'white')
        # ax.set_title(title, fontsize = plt.rcParams['font.size'])
        # ax.set_ylabel('Basic reproduction number')
        # ax.set_xlabel('Age group (weight)')
        sns.lineplot(df, ax = ax)
        ax.set_title(title)
        ax.legend(ncols = 1, fontsize = 'small', fancybox = False, )
        if label != '2A':
            ax.hlines(y=0, xmin = ax.get_xlim()[0], xmax = ax.get_xlim()[1], 
                      colors = 'grey', lw = 0.5)
    ax.set_xlabel(r'$\mathcal{R}_0$')
    ax.text(-0.15, 1.1, f'[{label}]', transform=ax.transAxes, fontsize=10, va='top')    
    ax.grid(True, which = 'both', lw = 0.5, ls = ':')
    ax.set_xlim(1, R_0_list[-1])
fnum = str(int(time.time()))
plt.savefig('Dilemma line plots'+suptitle+'.png', dpi = 400)
f.savefig('Dilemma line plots'+suptitle+'.pdf')
#%%
#df = pd.read_html('view.html')

# n = np.array([0.4, 0.6]) #Define the sizes of sub-populations
# no_g = len(n)
# gamma = 1 #Recovery rate
#A = np.array([1, 1, 1]) #Weights of the objective function that is minimized. [1, 1, 1] means the final size of the epidemic is minimized


# b1 = 1.2
# b2 = 2.0
# alpha = 1.5 #For parameter variation. B_12 = b1*b2*alpha, B21 = b1*b2*alpha, B11 = b1^2, B22 = b2^2

# B = np.array([[b1**2, b1*b2*alpha], [b2*b1*alpha, b2**2]])

# fs_og, fs_opt_numerical = numerical.Executor(n, B, gamma, A, True)
