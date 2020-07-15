#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:05:40 2020

@author: peter
"""

#A scratch to do decaying exponentials
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import FRI_functions as FRIF


import extract_exponentials as ee
import double_consistency_search as dc

def calculate_mean_offset(tk,tk_true):
    offsets = []
    for sp in tk:
        offsets.append(tk_true[np.argmin(np.abs(tk_true - sp))] - sp)
    return offsets


#6np.random.seed(0)

T = 1/8
lam = 0.5
tau = 0.5
length = 40
oversamp = 64

#Make long signal of decaying
tk_true,ak_true,t,x = FRIF.make_signal(length,(1/T)*oversamp,firing_rate = lam,tau = tau,spike_std = 0)


if False:
    #add rolling shutter 
    shutter_length = T*5
    
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

sc = 0.001*np.max(x)
noise = np.random.normal(scale = sc,size = len(x))
x += noise

SNR = 10*np.log10(np.sum(x**2)/np.sum(noise**2))
print(f'SNR: {SNR}')



plt.cla()
plt.plot(t,x)
for i in range(0,int(len(t)),2):
    plt.axvspan(t[i],t[(i+1)],facecolor='k', alpha=0.2)
    plt.plot([t[i]+T/2,t[i]+T/2],[0,8],'k',alpha = 0.1)
    plt.plot([t[i]-T/2,t[i]-T/2],[0,8],'k',alpha = 0.1)

tk1,ak1,_ = dc.double_consistency_histogram(x,t,tau,winlens = [32,8],shutter_length = None)
tk2,ak2,_ = dc.double_consistency_histogram(x,t,tau,winlens = [32,8],shutter_length = shutter_length)

plt.plot(tk_true,ak_true*0+8,'.g')
plt.plot(tk1,ak1*0+7,'.r')
plt.plot(tk2,ak2*0+6,'.k')

if len(tk1) > 0:
    cos_score1,_ = dc.compare_spike_trains(tk1,tk_true,1,1/T,tau)
    print(f'No box: {cos_score1}')

if len(tk2) > 0:
    cos_score2,_ = dc.compare_spike_trains(tk2,tk_true,1,1/T,tau)
    print(f'With box: {cos_score2}')
    
    
off2 = calculate_mean_offset(tk2,tk_true)
#plt.figure()
#plt.plot(off2) 
    
def get_mean_score(shutter,noise_level, repeats = 5):
    T = 1/10
    lam = 0.5
    tau = 0.5
    length = 40
    oversamp = 64
    
    scores = []
    scores_box = []
    SNRs = []
    for i in range(repeats):

        
        #Make long signal of decaying
        tk_true,ak_true,t,x = FRIF.make_signal(length,(1/T)*oversamp,firing_rate = lam,tau = tau,spike_std = 0)
        
        #add rolling shutter 
        shutter_length = T*shutter
        
        shutter_fcn = np.zeros(int(np.round(shutter_length*oversamp/T))+2)
        shutter_fcn[1:-1] = 1/int(np.round(shutter_length*oversamp/T))
        
        #filter with the shutter
        x = scipy.signal.convolve(x,shutter_fcn,mode = 'full')
        
        t = t[::oversamp]
        x = x[:-len(shutter_fcn) + 1:oversamp]
        
        sc = noise_level*np.max(x)
        noise = np.random.normal(scale = sc,size = len(x))
        x += noise
        
        SNR = 10*np.log10(np.sum(x**2)/np.sum(noise**2))
    
        tk1,ak1,_ = dc.double_consistency_histogram(x,t,tau,winlens = [32,8],shutter_length = None)
        tk2,ak2,_ = dc.double_consistency_histogram(x,t,tau,winlens = [32,8],shutter_length = shutter_length)
        
        if len(tk1) > 0:
            cos_score1,_ = dc.compare_spike_trains(tk1,tk_true,1,1/T,tau)
        else:
            cos_score1 = np.NaN
        
        if len(tk2) > 0:
            cos_score2,_ = dc.compare_spike_trains(tk2,tk_true,1,1/T,tau)
        else:
            cos_score2 = np.NaN
        
        SNRs.append(SNR)
        scores.append(cos_score1)
        scores_box.append(cos_score2)
        
        
    return np.nanmean(scores),np.nanmean(scores_box),np.nanmean(SNRs)

'''

noises = np.arange(0.001,0.5,0.05)
shutters = np.arange(1,10,1)

res = np.zeros((len(noises),len(shutters)))
res_box = np.zeros_like(res)
snrs = np.zeros_like(res)

for idx1,n in enumerate(noises):
    for idx2, s in enumerate(shutters):
        scores,scores_box,snr = get_mean_score(s,n,repeats = 100)
        res[idx1,idx2] = scores
        res_box[idx1,idx2] = scores_box
        snrs[idx1,idx2] = snr
        
np.save('./res.npy',res)
np.save('./res_box.npy',res_box)
np.save('./snrs.npy',snrs)
'''