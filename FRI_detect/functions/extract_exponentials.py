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
    t : 1D Float vector 
        Time stamps of samples in x.
    x : 1D float vector. Same length as t
        Fluorescent samples.
    phi : 1D Float vector.
        Exponential reproducing kernel.
    t_phi : 1D Float vector
        Time stamps of phi.
    tau : float
        Exponential coefficient of the fluorescent signal. e^(-t/tau)

    Returns
    -------
    z_n : 1D float vector - length len(x) - 1
        Equivalent to diracs sampled with phi*\beta_{\alpha T}.
    t_n : 1D float vector length of z_n
        Time stamps.

    '''
        
    T = np.mean(np.diff(t))
    
    #sample signal with exp. reproducing kernel
    y_n = T*scipy.signal.convolve(x,phi)
    t_y = np.linspace(t[0]+t_phi[0],t[-1]+t_phi[-1],len(y_n))
    
    #reduce to dirac sampling
    z_n = y_n[1:] - y_n[:-1]*np.exp(-T/tau)
    t_n = t_y[1:]

    return z_n,t_n


def window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec, fixed_K = None, taper_window = False):
    '''
    Extracts exponentials in a sliding window throughout the z_n signal

    Parameters
    ----------
    z_n : 1D float vec
        Finite difference signal derived from fluorescent samples.
    t_n : 1d float vec
        z_n time stamps.
    c_m_n : 2D array floats
        Exponential reproducing coefficients.
    n_vec : 1D array ints
        Vector of sample locations for the exponential reproduction.

    Returns
    -------
    all_tk : list of 1D float vectors
        List where ith element is a vector of spikes estimated from the ith window.
    all_ak : list of 1D float vectors
        Amplitudes of spikes in same format as all_tk.

    '''
    T = np.mean(np.diff(t_n))
    
    #get sliding window indices
    sliding_idx = n_vec[None,:] + np.arange(len(z_n) - len(n_vec) + 1)[:,None]
    z_n_window = z_n[sliding_idx]
    t_n_window = t_n[sliding_idx] 
    
    #apply a window function to remove border effects
    if taper_window:
        taper = scipy.signal.windows.tukey(len(n_vec),alpha = 0.5,sym = True)
        z_n_window *= taper[None,:]
        

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


def sliding_window_detect(t,x,tau,win_len,fixed_K = None, taper_window = False):
    '''
    A function that detects calcium transients (instantaneous rise, decay constant tau)
    from a fluorescent signal, x, using a sliding window of length win_len and 
    returns **all** the detected spikes from all the windows.

    Parameters
    ----------
    t : 1D array of floats
        Time stamps of x. Assumes even sampling.
    x : 1D array of floats. Same length as t.
        Fluorescent signal.
    tau : float
        Exponential decay constant e^(-t/tau).
    win_len : Even integer. 
        Window length for sliding window. Recommend  8 =< win_len =< 128, power of 2 typical.
        need to change with sampling rate.
    fixed_K : int, optional
        If you want the model order (num spikes in window) to be fixed set this as an int. The default is None.
        If is None model order is estimated from samples.
    taper_window : bool, optional
        If true will apply a window function to remove edge effects.
        Typically not required for regular applications.
        The default is False.

    Returns
    -------
    all_tk : list of 1D float vectors
        List where ith element is a vector of spikes estimated from the ith window.
    all_ak : list of 1D float vectors
        Amplitudes of spikes in same format as all_tk.

    '''
    #Get sampling period
    T = np.mean(np.diff(t))
    #Calculate the required sampling kernel, exponential reproducing coeffs, etc.
    phi,t_phi,c_m_n,n_vec,alpha_vec = ges.decaying_exp_filters(win_len, T, tau)
    #Convert the signal to equivalent sampled diracs
    z_n,t_n = convert_exponential_to_dirac(t,x,phi,t_phi,tau)
    #Extract in sliding window
    all_tk,all_ak = window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec,fixed_K=fixed_K,taper_window=taper_window)
    return all_tk,all_ak

def sliding_window_detect_box_filtered(t,x,tau,win_len,shutter_length,fixed_K = None, taper_window = True):
    '''
    A function that detects calcium transients (instantaneous rise, decay constant tau)
    from a fluorescent signal, x, which has been collected using an integrating detector 
    with integration period shutter_length
    using a sliding window of length win_len and 
    returns **all** the detected spikes from all the windows.

    Parameters
    ----------
    t : 1D array of floats
        Time stamps of x. Assumes even sampling.
    x : 1D array of floats. Same length as t.
        Fluorescent signal.
    tau : float
        Exponential decay constant e^(-t/tau).
    win_len : Even integer. 
        Window length for sliding window. Recommend  8 =< win_len =< 128, power of 2 typical.
        need to change with sampling rate.
    shutter_length : float
        Length of the integration period from the integrating detector.
    fixed_K : int, optional
        If you want the model order (num spikes in window) to be fixed set this as an int. The default is None.
        If is None model order is estimated from samples.
    taper_window : bool, optional
        If true will apply a window function to remove edge effects.
        Typically not required for regular applications.
        The default is False.

    Returns
    -------
    all_tk : list of 1D float vectors
        List where ith element is a vector of spikes estimated from the ith window.
    all_ak : list of 1D float vectors
        Amplitudes of spikes in same format as all_tk.

    '''
    #Get sampling period
    T = np.mean(np.diff(t))
    #Calculate the required sampling kernel, exponential reproducing coeffs, etc.
    phi,t_phi,c_m_n,n_vec,alpha_vec = ges.box_decaying_exp_filters(win_len, T, tau, shutter_length)
    #Convert the signal to equivalent sampled diracs
    z_n,t_n = convert_exponential_to_dirac(t,x,phi,t_phi,tau)
    #Extract in sliding window
    all_tk,all_ak = window_extract(z_n,t_n,c_m_n,n_vec,alpha_vec,fixed_K=fixed_K, taper_window = taper_window)
    return all_tk,all_ak
