#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:20:00 2020

@author: peter
"""
import numpy as np
import scipy.signal

from FRI_detect.functions import matrix_pencil as mp
from FRI_detect.functions import generate_e_spline as ges

def convert_exponential_to_dirac(t,x,phi,t_phi,tau):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    t_phi : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.

    Returns
    -------
    z_n : TYPE
        DESCRIPTION.
    t_n : TYPE
        DESCRIPTION.

    '''
        
    T = np.mean(np.diff(t))
    
    #sample signal with exp. reproducing kernel
    y_n = T*scipy.signal.convolve(x,phi)
    t_y = np.linspace(t[0]+t_phi[0],t[-1]+t_phi[-1],len(y_n))
    
    #reduce to dirac sampling
    z_n = y_n[1:] - y_n[:-1]*np.exp(-T/tau)
    t_n = t_y[1:]

    return z_n,t_n


def window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec, fixed_K = None):
    '''
    Extracts exponentials in a sliding window

    Parameters
    ----------
    z_n : TYPE
        DESCRIPTION.
    t_n : TYPE
        DESCRIPTION.
    c_m_n : TYPE
        DESCRIPTION.
    n_vec : TYPE
        DESCRIPTION.

    Returns
    -------
    all_tk : TYPE
        DESCRIPTION.
    all_ak : TYPE
        DESCRIPTION.

    '''
    T = np.mean(np.diff(t_n))
    
    #get sliding window indices
    sliding_idx = n_vec[None,:] + np.arange(len(z_n) - len(n_vec) + 1)[:,None]
    z_n_window = z_n[sliding_idx]
    t_n_window = t_n[sliding_idx] 

    #Calculate moments
    s_m_window = np.sum(c_m_n[:,None,:]*z_n_window[None,:,:],-1)
    s_m_window = np.moveaxis(s_m_window,-1,0)
    
    #Extract from each window
    all_tk = []
    all_ak = []
    for win_idx,win_sm in enumerate(s_m_window):
        tk,ak = mp.retrieve_tk_ak(win_sm, T, alpha_vec,K = fixed_K,thresh = 0.3)
        all_tk.append(tk + t_n_window[win_idx,0] + n_vec[-1]*T)
        all_ak.append(ak)
    
    return all_tk,all_ak


def sliding_window_detect(t,x,tau,win_len,fixed_K = None):
    '''
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    win_len : TYPE
        DESCRIPTION.
    fixed_K : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    all_tk : TYPE
        DESCRIPTION.
    all_ak : TYPE
        DESCRIPTION.

    '''
    T = np.mean(np.diff(t))
    phi,t_phi,c_m_n,n_vec,alpha_vec = ges.decaying_exp_filters(win_len, T, tau)
    z_n,t_n = convert_exponential_to_dirac(t,x,phi,t_phi,tau)
    all_tk,all_ak = window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec,fixed_K=fixed_K)
    return all_tk,all_ak