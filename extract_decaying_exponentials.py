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
        
    #locate diracs using matrix pencil
    P = int(len(n_vec)/2)
    u_k = matrix_pencil.acmp_p(s_m, K, int(np.round(P/2)), P, p = 1)
    
    tk = np.real(T * np.log(u_k) / lbda) #these are shifted by n[0]*T
    
    #find amplitudes ak
    A = np.zeros((K,K)).astype(np.complex128)
    for i in range(K):
        A[i,:] = u_k[:K]**i
        
    B = s_m[:K]
    b_k = scipy.linalg.solve(A,B)
    ak = np.real(b_k*np.exp(-alpha_0*tk/T))
    
    #shift to correct time
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