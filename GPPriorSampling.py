# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:28:17 2022

@author: Edwar
"""

import numpy as np
import matplotlib.pyplot as plt

def Gaussian_kernel(x,y):
    sig = 1
    l = 0.1
    return sig**2*np.exp(-(x-y)**2/(2*l))

def Linear_kernel(x,y):
    sig_b = 1
    sig = 1
    c = 0
    return sig_b**2+sig**2*(x-c)*(y-c)

def Periodic_kernel(x,y):
    sig = 1
    l = 1
    p = 0.3
    return sig**2*np.exp(-(2*np.sin((np.pi*abs(x-y))/p)**2/(l**2)))

def Linear_plus_periodic(x,y):
    return Linear_kernel(x,y)+Periodic_kernel(x,y)
    

def get_covariance(kernel,mesh_count):
    c = np.zeros((mesh_count,mesh_count))
    mesh_size = 1/(mesh_count-1)
    for i in range(mesh_count):
        x = i*mesh_size
        for j in range(mesh_count):
            y = j*mesh_size
            c[i,j] = kernel(x,y)
    return c

kernels = [Gaussian_kernel,Linear_kernel,Periodic_kernel,Linear_plus_periodic]

mesh_count = 101
mesh = np.linspace(0,1,mesh_count)

for kernel in kernels:
    covar = get_covariance(kernel,mesh_count)
    sample = np.random.multivariate_normal([0]*mesh_count,covar)
    print(sample[0])
    plt.plot(mesh,sample,label = kernel.__name__.replace("_"," "))
    
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian Process Prior Samples")
plt.legend()

    