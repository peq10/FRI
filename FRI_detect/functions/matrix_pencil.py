#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:02:11 2020

@author: peter
"""
import scipy.linalg
import numpy as np

def matrix_pencil(sm,K = None, thresh = 0.3):
    '''
    Noisy matrix pencil method for retrieving uk from sample moments (see 
    http://www.commsp.ee.ic.ac.uk/~pld/group/PhDThesis_Onativia15.pdf p.50)
    
    Always uses Splits the samples in the middle although that may change in future. 
    Can estimate number of diracs.

    Parameters
    ----------
    sm : complex 1d array
        Signal moments.
    K : Int, optional
        Number of diracs in the signal. The default is None.
        Is estimated if None is given.
    thresh : float, optional
        Threshold for estimating K from signal moments. The default is 0.3.

    Returns
    -------
    uu_k : Complex array length K
        Values from which tk and ak can be calculated.

    '''
    
    M = np.ceil(len(sm)/2).astype(int)
    S = scipy.linalg.toeplitz(sm[M:],sm[M::-1])
    U,s,_, = scipy.linalg.svd(S)    
    
    if K is None:
        
        if s.max() == 0:
            return np.array([])
        else:
            s = s/s.max()
            K = np.sum(s>thresh)
    
    U = U[:,:K]

    S0 = U[1:,:]
    S1 = U[:-1,:]
    Z = np.matmul(scipy.linalg.pinv(S1),S0)
    
    uu_k = scipy.linalg.eig(Z)[0]
    
    return uu_k

def matrix_pencil_noiseless(sm,K):
    '''
    Retrieves Uk in noiseless case using matrix pencil

    Parameters
    ----------
    sm : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.

    Returns
    -------
    uu_k : TYPE
        DESCRIPTION.

    '''
    S = scipy.linalg.toeplitz(sm[K-1:],sm[K-1::-1])
    S0 = S[1:,:]
    S1 = S[:-1,:]
    Z = np.matmul(scipy.linalg.inv(S1),S0)
    uu_k = scipy.linalg.eig(Z)[0]
    return uu_k

def retrieve_tk_ak(sm,T,alpha_vec, K = None,thresh = 0.3, remove_negative = True):
    '''
    

    Parameters
    ----------
    sm : 1D array of signal moments
        DESCRIPTION.
    T : Float
        Sampling period.
    alpha_vec : 1D complex array
        vector of sampling kernel exponential reproductions.
    K : int, optional
        Number of diracs in sample. The default is None.

    Returns
    -------
    t_k : 1D array of real floats
        Estimated dirac times.
    a_k : 1D array of real floats
        Estimated dirac sizes.

    '''
    
    uk = matrix_pencil(sm,K = K,thresh = thresh)    
    
    lbda = np.mean(np.real(np.diff(alpha_vec)/1j))
    omega_0 = np.real(alpha_vec[0]/1j)
    K = len(uk)
    
    t_k = np.real(np.log(uk)*T/(1j*lbda))
    A = np.zeros((K,K)).astype(np.complex128)
    for i in range(K):
        A[i,:] = uk**i
    B = sm[:K]
    b_k = scipy.linalg.solve(A,B)
    a_k = np.real(b_k*np.exp(-1j*omega_0*t_k/T))
    
    if remove_negative:
        pos = a_k > 0
        a_k = a_k[pos]
        t_k = t_k[pos]

    return t_k,a_k




if __name__ == '__main__':
    pass