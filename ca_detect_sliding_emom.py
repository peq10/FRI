#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:45:39 2020

@author: peter
"""
import numpy as np
import extract_decaying_exponentials as ede
import generate_e_spline as ges
import get_c_mn_exp as gce
import scipy.special

def sliding_window_detect(x,t,win_len,tau, mode = 'estimate', fixed_K = None):
    '''
    I only allow  diff(t) == T

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    win_len : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 'estimate'.

    Returns
    -------
    tk : TYPE
        DESCRIPTION.
    ak : TYPE
        DESCRIPTION.
    win_idx : TYPE
        DESCRIPTION.
    K_i : TYPE
        DESCRIPTION.

    '''
    T = np.mean(np.diff(t))
    
    #construct sampling kernel
    over_samp = 512
    N = win_len
    P = N/2
    m = np.arange(P+1)
    alpha_0 = -1j * np.pi / 2
    lbda = 1j * np.pi / P
    alpha_vec = alpha_0 + lbda*m
    
    #for some reason we don't actually use phi, just the time stamps??
    phi, t_phi = ges.generate_e_spline(alpha_vec, T/over_samp, T = T)
    t_diric = np.arange(-(P+1)/2,(P+1)/2 +1/over_samp,1/over_samp)*(2*np.pi)/(P+1)
    phi = scipy.special.diric(t_diric, P+1)
    phi = phi.real
        
    #generate ps
    alpha = 1/tau
    beta_alpha_t, t_beta = ges.generate_e_spline(np.array([-alpha*T]), 1/over_samp)
    beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))
    psi = (T/over_samp)*scipy.signal.convolve(phi,beta_alpha_t)
    t_psi = np.arange(len(psi))*T/512
    
    #get cmn coefficients
    if N%2 == 0:
        n_vec = np.arange(-int(N/2),int(N/2))
    else:
        n_vec = np.arange(-int((N-1)/2),int((N+1)/2))
        

    c_m_n = gce.get_c_mn_exp2(alpha_vec,n_vec,psi,t_psi,T = T)
    #now downsample phi, t_phi to remove oversamplign
    phi = phi[::over_samp]
    t_phi = t_phi[::over_samp]
    
    #iterate through the vector x and detect expoentials in sliding window
    K_i = np.zeros(len(x) - win_len)
    all_tk = []
    all_ak = []
    win_idx = []
    
    for i_0 in range(len(x) - win_len):
        x_part = x[i_0:i_0+win_len]
        t_part = t[i_0:i_0+win_len]
        
        if mode == 'estimate':
            tk,ak = ede.extract_decaying_exponentials(x_part, t_part, 
                                                      tau, phi, t_phi,
                                                      alpha_0, lbda, T,
                                                      c_m_n, n_vec, K = None)
        elif mode == 'fixed':
            if fixed_K is None:
                raise ValueError('Must provide fixed_K for mode == fixed')
            tk,ak = ede.extract_decaying_exponentials(x_part, t_part, 
                                                      tau, phi, t_phi,
                                                      alpha_0, lbda, T,
                                                      c_m_n, n_vec, K = fixed_K)
            
        #remove negative spikes?
        pos_sp = ak >= 0
        tk = tk[pos_sp]
        ak = ak[pos_sp]
    
        all_tk.append(tk)
        all_ak.append(ak)
        win_idx.append(i_0*np.ones(len(tk)))
        K_i[i_0] = len(tk)
    
    
    
    return np.concatenate(all_tk),np.concatenate(all_ak),np.concatenate(win_idx).astype(int),K_i.astype(int)
    

def sliding_window_detect_box_filtered(x,t,win_len,tau,shutter_length, mode = 'estimate', fixed_K = None):
    '''
    I only allow  diff(t) == T

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    win_len : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 'estimate'.

    Returns
    -------
    tk : TYPE
        DESCRIPTION.
    ak : TYPE
        DESCRIPTION.
    win_idx : TYPE
        DESCRIPTION.
    K_i : TYPE
        DESCRIPTION.

    '''
    T = np.mean(np.diff(t))
    
    #construct sampling kernel
    over_samp = 512
    N = win_len
    P = N/2
    m = np.arange(P+1)
    alpha_0 = -1j * np.pi / 2
    lbda = 1j * np.pi / P
    alpha_vec = alpha_0 + lbda*m
    
    #for some reason we don't actually use phi, just the time stamps??
    phi, t_phi = ges.generate_e_spline(alpha_vec, T/over_samp, T = T)
    t_diric = np.arange(-(P+1)/2,(P+1)/2 +1/over_samp,1/over_samp)*(2*np.pi)/(P+1)
    phi = scipy.special.diric(t_diric, P+1)
    phi = phi.real
        
    #generate psi
    alpha = 1/tau
    beta_alpha_t, t_beta = ges.generate_e_spline(np.array([-alpha*T]), 1/over_samp)
    beta_alpha_t = np.concatenate(([0],beta_alpha_t[:0:-1]))
    psi = (T/over_samp)*scipy.signal.convolve(phi,beta_alpha_t)

    #add in the effect of the rolling shutter filter
    shutter_fcn = np.zeros(int(np.round(shutter_length/np.mean(np.diff(t_phi))))+2)
    shutter_fcn[1:-1] = 1
    shutter_fcn = shutter_fcn / np.sum(shutter_fcn)
    #convolve psi with the shutter fcn
    psi = scipy.signal.convolve(psi,shutter_fcn)
    
    t_psi = np.arange(len(psi))*T/512
    
    #get cmn coefficients
    if N%2 == 0:
        n_vec = np.arange(-int(N/2),int(N/2))
    else:
        n_vec = np.arange(-int((N-1)/2),int((N+1)/2))
        

    c_m_n = gce.get_c_mn_exp2(alpha_vec,n_vec,psi,t_psi,T = T)
    #now downsample phi, t_phi to remove oversamplign
    phi = phi[::over_samp]
    t_phi = t_phi[::over_samp]
    
    #iterate through the vector x and detect expoentials in sliding window
    K_i = np.zeros(len(x) - win_len)
    all_tk = []
    all_ak = []
    win_idx = []
    
    for i_0 in range(len(x) - win_len):
        x_part = x[i_0:i_0+win_len]
        t_part = t[i_0:i_0+win_len]
        
        if mode == 'estimate':
            tk,ak = ede.extract_decaying_exponentials(x_part, t_part, 
                                                      tau, phi, t_phi,
                                                      alpha_0, lbda, T,
                                                      c_m_n, n_vec, K = None)
        elif mode == 'fixed':
            if fixed_K is None:
                raise ValueError('Must provide fixed_K for mode == fixed')
            tk,ak = ede.extract_decaying_exponentials(x_part, t_part, 
                                                      tau, phi, t_phi,
                                                      alpha_0, lbda, T,
                                                      c_m_n, n_vec, K = fixed_K)
    
        #remove negative spikes?
        pos_sp = ak >= 0
        tk = tk[pos_sp]
        ak = ak[pos_sp]
        
        all_tk.append(tk)
        all_ak.append(ak)
        win_idx.append(i_0*np.ones(len(tk)))
        K_i[i_0] = len(tk)
    
    
    
    
    
    return np.concatenate(all_tk),np.concatenate(all_ak),np.concatenate(win_idx).astype(int),K_i.astype(int)