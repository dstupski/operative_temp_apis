#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:15:24 2019

@author: davidstupski
"""
from __future__ import division
import pandas as pd
import numpy as np
import uncertainties as u
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
import matplotlib.pyplot as plt
nsim = 10000
np.random.seed(1)
#hk0  = np.random.normal(.0030, .000415, nsim)
hk1 = np.random.normal(.0049, .000680, nsim)
a_b = np.random.normal(.903, .0159, nsim)
a_r = np.random.normal(.8735, .021, nsim)
a_surf = np.random.normal(.0000867*.85, .00000766, nsim)
m = np.random.normal(.046, .002, nsim)
evap_err = np.random.normal(0, 25.3, nsim)
met_err = np.random.normal(0, 56, nsim)
x = []
x_c = []
x_other = []
x_other_ = []
std_vector = []
std_vector_c = []
mean_ = []
mean_c = []
#print len(x)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 4) )
plt.style.use("seaborn-white")
plt.tight_layout()
for hk1, a_b, a_r, a_surf, m, evap_err, met_err in zip(hk1, a_b, a_r, a_surf, m, evap_err, met_err):
    def heat_transfer_m(Tb, t, i, Ta):
        c= 4.18
        epsilon = .96
        esky=9.2*(10**-6)*(Ta+273)**2
        sigma = 5.67*10**-8 
        #1.63 constant arises from the heat production per unit mass in the original stephens and roberts paper the total mass over the effective thermal mass.  Mass term in denominator dissapeares since data are in W/g multipying by 1/c is W/(g*c)= dT/dt
        dTbdt = -1.0 * hk1*(Tb-Ta)/(c*m) + a_b *a_surf*.25*(i)/(c*m)+a_r*a_surf*.5*i*.25/(c*m)+100*a_b*a_surf*.5/(c*m) - sigma*a_surf*(epsilon*(Tb+273)**4-.5*esky*(Ta+273)**4-.5*.96*(Ta+273)**4)/(c*m) + .0753/m*((-29.83514*Tb+1763.0566+met_err)/(1000))/(c) - .0753/m*(.0022*2.71828**(.24413*Tb)+evap_err)/(1000)/(c)
        return dTbdt
    T0= 35.9
    ntime = 200
    t_ = np.linspace(0, 200, ntime)
    sol_ = odeint(heat_transfer_m, T0, t_, args = (800, 30))
    sol__ = odeint(heat_transfer_m, T0, t_, args = (600, 18))
    #sol_c = odeint(heat_transfer_m, T0, t_, args = (600, 10))
    #print sol
    sol_list = sol_.tolist()
    ax[0].plot(t_, sol_, color = "indianred", alpha = 0.005)
    ax[0].plot(t_, sol__, color = "lightblue", alpha = .005)
    sol_clist = sol__.tolist()
    for i in range(len(sol_list)):
        x.append(sol_list[i][0])
    for i in range(len(sol_list)):
        x_other.append(sol_list[-1][0])
    for i in range(len(sol_clist)):
        x_c.append(sol_clist[i][0])
    for i in range(len(sol_clist)):
        x_other_.append(sol_clist[-1][0])
    #print(sol_list)
    #print sol_list
    #for i in range(len(sol_list)):
    #    x.append(sol_list[i][0])
        #val = sol_list[i][0]       
        #x[i] += sol_list[i][0]
#tau = np.linspace(0,200,200)
#ax.plot(tau, x, color = "k")
for i in range(ntime):
    temp_vector = []
    for j in range(nsim):
        z = j*ntime+i
        #print z
        temp_vector.append(x[z])
    #print temp_vector
    std_vector.append(np.std(temp_vector))
    mean_.append(np.mean(temp_vector))
for i in range(ntime):
    temp_vector = []
    for j in range(nsim):
        z = j*ntime+i
        #print z
        temp_vector.append(x_c[z])
    #print temp_vector
    std_vector_c.append(np.std(temp_vector))
    mean_c.append(np.mean(temp_vector))
#print len(std_vector)        

#initiate numpy array for how many "seconds" we allow the model to run for
t = np.linspace(0,200,200)
#initial temperature of bee, here set to hive temperature
T0= 35.9
#mean bee paramaters + hot july conditions


#print solmax_vals[-1]
#print solmin_vals[-1]
#print std_vector[-1]
#print std_vector_c[-1]
print np.mean(x_other)
print np.std(x_other)
print np.mean(x_other_)
print np.std(x_other_)
      
ax[0].spines["right"].set_visible(False)
ax[0].spines["top"].set_visible(False)
ax[0].plot(t, mean_,color = "k", linestyle = "-")
#ax.plot(t, solmax, color = "green", linestyle = "")
#ax.plot(t, solmin, color = "yellow", linestyle = "")
#ax.fill_between(t, new_df["Hot Upper"], new_df["Hot Lower"], color = "red", alpha = '0.2')
ax[0].plot(t, mean_c, color = "k", linestyle = "--" )
#ax.plot(t, solcoldmax, color = "green", linestyle = "")
#ax.fill_between(t, new_df["Cold Upper"], new_df["Cold Lower"], color = "blue", alpha = '0.2')
ax[0].set_ylim(20, 52)
ax[0].set_xlim(0, 180)
ax[0].legend(loc = "upper right")
ax[0].set_xlabel("Time (s)", color = "k")
#ax.set_ylabel("Internal Temperature (C)", color = "k")
#ax.set_xlabel("Time (s)")
ax[0].set_ylabel("Temperature (C)")
ax[1] = sns.distplot(x_other, color = "indianred", norm_hist = True, vertical = True)
ax[1] = sns.distplot(x_other_, color = "lightblue",  norm_hist = True,  vertical = True) 
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
#ax[1].spines["left"].set_visible(False)
ax[1].axhline(y= np.mean(x_other), color = 'k', linestyle = "-", label = "Warm Day")
ax[1].axhline(y= np.mean(x_other_), color = 'k', linestyle = '--', label ="Cool Day")
ax[1].set_xlabel("Output Density")
#ax[1].set_ylabel("Output Density")
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
ax[1].set_ylim(20, 52)
plt.legend()
print " - 15% Surface Area"
#plt.title(")
#plt.savefig("/Users/davidstupski/Desktop/Monte_double_fig_hot_30_800_cold_18_600_4ms_correct_emiss.png")