#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:47:41 2020

@author: peter
"""

#sliding window dirac detect
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


import FRI_detect.functions.generate_e_spline as ges
import FRI_detect.functions.extract_exponentials as ee
import FRI_detect.functions.simulate_spike_train as sst
np.random.seed(0)

oversamp = 64
length = 30
lam = 0.5
T = 1/10
T_s = T/oversamp

noise_level = 0.0
tau = 0.5

#Make long signal of decaying
t_k,a_k,t,x = sst.make_signal(length,(1/T)*oversamp,firing_rate = lam,tau = tau,spike_std = 0)


if True:
    #add rolling shutter 
    shutter_length = T*10
    
    shutter_fcn = np.zeros(int(np.round(shutter_length*oversamp/T))+2)
    shutter_fcn[1:-1] = 1/int(np.round(shutter_length*oversamp/T))
    
    #filter with the shutter
    x = scipy.signal.convolve(x,shutter_fcn,mode = 'full')
    
    t = t[::oversamp]
    x = x[:-len(shutter_fcn) + 1:oversamp]
else:
    shutter_length = T
    t = t[::oversamp]
    x = x[::oversamp]

#plt.figure()
#plt.plot(t,x)
taper = True
win_len = 102
phi,t_phi,c_m_n,n_vec,alpha_vec = ges.box_decaying_exp_filters(win_len, T, tau, shutter_length)
z_n,t_n = ee.convert_exponential_to_dirac(t,x,phi,t_phi,tau)
all_tk1,all_ak1 = ee.window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec,fixed_K=None,taper_window = taper)

win_len2 = 32
phi,t_phi,c_m_n,n_vec,alpha_vec =  ges.box_decaying_exp_filters(win_len2, T, tau, shutter_length)
z_n,t_n = ee.convert_exponential_to_dirac(t,x,phi,t_phi,tau)
all_tk2,all_ak2 = ee.window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec,fixed_K=1,taper_window = taper)
    
#plot 
#fig,ax = plt.subplots()
plt.cla()
#plt.plot(t,-1*x*10)

for idx in range(len(all_tk1)):
    plt.plot(all_tk1[idx],np.ones(len(all_tk1[idx]))+idx,'.k')

for idx in range(len(all_tk2)):
    plt.plot(all_tk2[idx],np.ones(len(all_tk2[idx]))+idx + len(all_tk1),'.k')

idx += len(all_tk1)

for ti in t_k:
    plt.plot([ti,ti],[0,idx],'r')
    #plt.plot([ti+T/2,ti+T],[0,idx],'-r',alpha = 0.5)
    
for i in range(0,int(len(t)),2):
    plt.axvspan(t[i],t[(i+1)],facecolor='k', alpha=0.2)
    plt.plot([t[i]+T/2,t[i]+T/2],[0,idx],'k',alpha = 0.1)
    plt.plot([t[i]-T/2,t[i]-T/2],[0,idx],'k',alpha = 0.1)
    
spike_thresh = 0
all_tk = [np.concatenate(all_tk1),np.concatenate(all_tk2)]
all_ak = [np.concatenate(all_ak1),np.concatenate(all_ak2)]

#remove spikes below certain size?
for idx in range(len(all_ak)):
    keep = all_ak[idx] > spike_thresh
    all_ak[idx] = all_ak[idx][keep]
    all_tk[idx] = all_tk[idx][keep]


#generate histogram
hist_res = 1
bins = np.linspace(t[1],t[-1],int(len(t)*hist_res))

all_hists = []

for idx,tk in enumerate(all_tk):
    hist,_ = np.histogram(tk,bins = bins,density = True)
    all_hists.append(hist)

all_hists = np.array(all_hists)

jhist = all_hists[0]*all_hists[1]

thresh = 0.1
peaks = scipy.signal.find_peaks(jhist/np.nanmax(jhist) > thresh)[0]
sp_t = bins[peaks] + np.mean(np.diff(bins))/2
delta = np.mean(np.diff(bins))
precision = 1
amplitudes = np.zeros(len(sp_t))
for idx,ti in enumerate(sp_t):
    sp_tk = np.argwhere(np.abs(all_tk[0] - ti) < precision*delta)
    sp_ak = np.mean(all_ak[0][sp_tk.ravel()])
    amplitudes[idx] = sp_ak
    
for ti in sp_t:
    plt.plot([ti,ti],[0,len(all_tk1) + len(all_tk2)],'--g')


plt.plot(t_k,np.ones_like(t_k) + len(all_tk1) + len(all_tk2),'.r')
plt.plot(sp_t,np.ones_like(sp_t) + len(all_tk1) + len(all_tk2) + 10,'.g')
plt.plot(t,x*10)

#score,_ =  dc.compare_spike_trains(sp_t,t_k,noise_level,1/T,tau)
#print(score)