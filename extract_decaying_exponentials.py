#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:36:39 2020

@author: peter
"""
import numpy as np
import scipy.io
import scipy.linalg
import matrix_pencil

def extract_decaying_exponentials(x,t_x,tau,phi,t_phi,alpha_0,lbda,T,c_m_n,n_vec,K = None):
    '''
    Extracts the location and size of decaying exponentials from time course x
    using the FRI method.
    Adapted from Onativia's  original matlab code.
    For reference see http://www.commsp.ee.ic.ac.uk/~pld/group/PhDThesis_Onativia15.pdf pp.74 - 77.

    Parameters
    ----------
    x : float array length N
        Samples including decaying exponentials.
    t_x : float array length N
        Time stamps of x.
    tau : float
        The decay constant exp(-t/tau) of the exponentials in x.
    phi : float array length k
        Exponential reproducing kernel. Is Dirichlet kernel.
    t_phi : float array length k
        phi time stamps.
    alpha_0 : complex float
        The first exponent from the exponential reproducing kernel.
    lbda : complex float
        the increment between the exponents in the exponential reproducing kernel.
    T : Float
        Sampling period.
    c_m_n : M by N complex array
        Exponential reproducing coefficients.
    n_vec : N integer array
        n indices where c_m_n have been calculated.
    K : int, optional
        Number of exponentials in x. The default is None - if this is the case,
        will estimate the number.

    Returns
    -------
    tk : K float array
        estimated exponential start locs.
    ak : K float array
        estimated exponential sizes.

    '''
    
    
    #sampling period. this is just T in all cases as far as I can tekk
    t_s = t_x[1] - t_x[0]
    
    #now make x such that it is sampled by exponential repro.
    #I don't really know why its a circular convolution as opposed to normal convolution.
    y_n = t_s*circular_convolution(x,phi)

    z_n = y_n[1:] - y_n[:-1]*np.exp(-T/tau)

    #compute signal moments
    s_m = np.sum(c_m_n[:,1:]*z_n[None,:],-1)
    
    #estimate K if not provided
    if K is None:
        K = estimate_K(s_m)
        if K == 0:
            return np.array([]),np.array([])
        
    #locate diracs using matrix pencil
    P = int(len(n_vec)/2)
    u_k = matrix_pencil.acmp_p(s_m, K, int(np.round(P/2)), P, p = 1)
    
    tk = np.real(T * np.log(u_k) / lbda) #these are shifted by n[0]*T
    
    #find amplitudes ak - solving for bk eqn 2.57, p.45
    A = np.zeros((K,K)).astype(np.complex128)
    for i in range(K):
        A[i,:] = u_k[:K]**i
        
    B = s_m[:K]
    b_k = scipy.linalg.solve(A,B)
    ak = np.real(b_k*np.exp(-alpha_0*tk/T))
    
    #shift to correct time - undoing the effect of sampling with exponential repro kernel.
    tk -= n_vec[0]*T
    return tk,ak

def estimate_K(sm):
    kk = int(np.floor(len(sm)/2))
    S = scipy.linalg.toeplitz(sm[kk:],sm[kk::-1])  
    _,D,_ = scipy.linalg.svd(S)
    y_i = D/D[0]   
    #Estimate K thresholding the eigenvalues
    K = np.sum(y_i > 0.3)   
    K = min([K,kk-1])
    return K

def circular_convolution(a,b):
    return np.convolve(np.concatenate((a,a)),b)[len(b)-1:len(b)-1+len(a)]
    
    
def load_input():
    i = np.load('./data/dec_input.npy',allow_pickle = True).item()
    x = i['x']
    t_x = i['t_x']
    alpha_vec = i['alpha_vec']
    phi = i['phi']
    t_phi = i['t_phi']
    alpha_0 = i['alpha_0']
    lbda = i['lbda']
    T = i['T']
    c_m_n = i['c_m_n']
    n_vec = i['n_vec']
    K = i['K']
    tau = i['tau']
    return x,t_x,alpha_vec,phi,t_phi,alpha_0,lbda,T,c_m_n,n_vec,K,tau
    

def test():
    x,t_x,alpha_vec,phi,t_phi,alpha_0,lbda,T,c_m_n,n_vec,K,tau = load_input()
    
    
    t_s = t_x[1] - t_x[0]
    
    #now make x such that it is sampled by exponential repro.
    #I don't really know why its a circular convolution as opposed to normal convolution.
    y_n = t_s*circular_convolution(x,phi)
    z_n = y_n[1:] - y_n[:-1]*np.exp(-T/tau)
    
    #compute signal moments
    s_m = np.sum(c_m_n[:,1:]*z_n[None,:],-1)
    
    sm_test = scipy.io.loadmat('./data/sm.mat')  
    s_m_test = np.squeeze(sm_test['s_m'])
    
    np.testing.assert_allclose(s_m,s_m_test)
    np.save('./data/sm.npy',s_m)
    kk = int(np.floor(len(s_m)/2))
    S = scipy.linalg.toeplitz(s_m[kk:],s_m[kk::-1])
    
    S_test = np.squeeze(scipy.io.loadmat('./data/S.mat')['S'])
    
    np.testing.assert_allclose(S,S_test)
    
    _,D,_ = scipy.linalg.svd(S)
    y_i = D/D[0]
    
    #Estimate K thresholding the eigenvalues
    K = np.sum(y_i > 0.3)
    
    K = min([K,kk-1])
    
    #locate diracs using matrix pencil
    P = int(len(n_vec)/2)
    u_k = matrix_pencil.acmp_p(s_m, K, int(np.round(P/2)), P, p = 1)
    tk = np.real(T * np.log(u_k) / lbda) #these are shifted by n[0]
    
    #find amplitudes ak
    A = np.zeros((K,K)).astype(np.complex128)
    for i in range(K):
        A[i,:] = u_k[:K]**i
        
    B = s_m[:K]
    b_k = scipy.linalg.solve(A,B)
    ak = np.real(b_k*np.exp(-alpha_0*tk/T))
    
    
    #shift to correct time
    tk -= n_vec[0]*T
    
    corr_tk = [0.7096,2.4251]
    corr_ak = [3.5199,3.24]
    
    np.testing.assert_allclose(np.round(tk,4),corr_tk)
    np.testing.assert_allclose(np.round(ak,4),corr_ak)
    
    
def test2():
    x,t_x,alpha_vec,phi,t_phi,alpha_0,lbda,T,c_m_n,n_vec,K,tau = load_input()
    tk,ak = extract_decaying_exponentials(x,t_x,tau,phi,t_phi,alpha_0,lbda,T,c_m_n,n_vec)
    corr_tk = [0.7096,2.4251]
    corr_ak = [3.5199,3.24]
    np.testing.assert_allclose(np.round(tk,4),corr_tk)
    np.testing.assert_allclose(np.round(ak,4),corr_ak)
    
if __name__ == '__main__':
    test()
    test2()