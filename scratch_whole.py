#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:56:07 2020

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import FRI_functions as FRIF


# get the initial state of the RNG


fs = 10
length = 5
tau = 0.5
win_len = 5

tk_true,ak_true,t,x = FRIF.make_signal(length,fs,tau = tau)

plt.plot(t,x)
plt.show()

#first filter with an exponential reproducing kernel
#generate kernel
over_samp = 512
N = win_len
P = N/2
m = np.arange(P+1)
alpha_0 = -1j * np.pi / 2
lbda = 1j * np.pi / P
alpha_vec = alpha_0 + lbda*m

phi, t_phi = FRIF.generate_e_spline.generate_e_spline(alpha_vec, 1/fs/over_samp)