#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:04:07 2020

@author: peter
"""
import numpy as np
        
def get_c_mn_exp2(alpha_m, n, phi, t_phi, T = 1):
    '''
    
    Calculates the c_{m,n} coefficients for reproduction of exponentials (coefficients alpha_m)
    using the kernel phi shifted by the integer sample points given by n.

    For technical details see https://doi.org/10.25560/49792 Appendix A.2

    Parameters
    ----------
    alpha_m : 1D complex (purely imaginary) float vector
        The exponentials to be reproduced.
    n : 1d array of ints
        The kernel shift positions.
    phi : 1d complex float vec
        The E-spline kernel.
    t_phi : 1d float vec
        Kernel time points.
    T : float, optional
        Sampling period. The default is 1.

    Returns
    -------
    c_m_n : 2D complex array shape (len(alpha_vec),len(n))
        The exponential reproducing coefficients.

    '''

    t_phi_sampling = np.mean(np.diff(t_phi))

    #get kernel boundaries
    t_1 = t_phi[0]/T
    t_2 = t_phi[-1]/T
    
    sta = np.ceil(np.round(-1*t_2,decimals = 3)).astype(int)
    sto = np.floor(np.round(-1*t_1,decimals = 3)).astype(int)
    npoints = sto - sta + 1
    #compute C_m_0 - I don't understand why the index here is not the full P?
    #time span of the t_phi vector without scaling?    
    l = np.linspace(sta,sto,npoints)
    idx = np.round(-1*T*(t_1 +l)/t_phi_sampling).astype(int)
    
    #calculate c_{m,0}
    phi_l = phi[idx]
    num = np.exp(alpha_m * 0)
    den = np.sum(np.exp(alpha_m[:,None] * l[None,:])*phi_l[None,:],-1)
    c_m_0 = num/den
    
    #calculate rest from cm0
    exp_mat = np.exp(alpha_m[:,None]*n[None,:])
    c_m_n = exp_mat*c_m_0[:,None]
    
    return c_m_n

