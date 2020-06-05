#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script taking it back to basics - use FRI matrix pencil to retrieve location of noiseless diracs

Created on Fri Jun  5 15:20:02 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF
import scipy.signal
import scipy.linalg
import scipy.special

#np.random.seed(0)


fs = 10
length = 10
rate = 0.5

#generate sampling kernel
over_samp = 512
N = 8
P = N/2
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda*m

phi, t_phi = FRIF.generate_e_spline(alpha_vec, 1/fs/over_samp)

# generate sampled diracs

tk_true, ak_true, t,x  = FRIF.make_delta_signal(length, fs, phi, rate)

plt.plot(t,x)
plt.stem(tk_true + t_phi.max()/2,ak_true,use_line_collection = True)

#get C_m_n

c_m_n = FRIF.get_c_mn_exp2(alpha_vec, np.arange(-int(N/2),int(N/2)), phi, t_phi)