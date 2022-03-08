# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:17:12 2021

@author: Edwar
"""

import numpy as np
import matplotlib.pyplot as plt

def sample_deep_gaussian(L,mesh_size):
    sample = np.random.multivariate_normal([0]*int(round(1/mesh_size)+1),compute_prior_covariance(np.linspace(0,1,int(round(1/mesh_size)+1))))
    u = sample_to_func(sample, mesh_size)
    u_1 = sample_to_func(sample, mesh_size)
    #plt.plot([i/len(sample-1) for i in range(len(sample))],sample,label=0)
    for i in range(1,L):
        covar = compute_covar(u,mesh_size,version="covar_func")
        sample = draw_sample(covar)
        u_2 = sample_to_func(sample,mesh_size)
    return u_1,u_2
        
def sample_to_func(sample,mesh_size):
    def u(x):
        return sample[int(round(x/mesh_size))]
    return u
           
def compute_covar(u,mesh_size,version="identity"):
    disc_count = int(round(1/mesh_size)+1)
    if version == "identity":
        return np.eye(disc_count)
    elif version == "covar_func":
        #Note: code written for 1-d regression
        sigma = 1
        c = np.zeros((disc_count,disc_count))
        for i in range(disc_count):
            for j in range(disc_count):
                x = i*mesh_size
                x_p = j*mesh_size
                #study pre_fac
                pre_fac = sigma**2*2**0.5*(F(u(x))*F(u(x_p)))**0.25/(F(u(x))+F(u(x_p)))**0.5
                c[i,j] = pre_fac*rho_s((2*abs(x-x_p))/(F(u(x))+F(u(x_p)))**0.5)
        return c
    
def kernel(x,y):
    eps = 10**(-12)
    return 1*np.exp(-abs(x-y)**2/(2*0.5**2))+eps*(abs(x-y)<0.00005)

def compute_prior_covariance(points):
    N = len(points)
    covar = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            covar[i][j] = kernel(points[i],points[j])
    return covar
    
def F(x):
    return x**2 #or exp

def rho_s(x):
    return np.exp(-x**2)
    
def draw_sample(covar):
    eps = 10**(-10)
    stabilizer = eps*(np.eye(covar.shape[0]))
    return np.random.multivariate_normal([0]*covar.shape[0],covar+stabilizer)

def main():
    N = 3
    L = 2 #fixed L=2
    mesh_size = 0.01
    mesh = np.linspace(0,1,round(1/mesh_size+1))
    samples = []
    for i in range(N):
        samples.append(sample_deep_gaussian(L,mesh_size))
    fig,ax = plt.subplots(1,3)
    for j in range(N):
        ax[j].plot(mesh,[samples[j][0](x) for x in mesh],label = "u\u2081")
        ax[j].plot(mesh,[samples[j][1](x) for x in mesh],label = "u\u2082")
        ax[j].set(xlabel="x",ylabel="y",xlim=(0,1),ylim=(-2,2))
    plt.tight_layout()
    fig.suptitle("Deep Gaussian Process Prior Samples",y=1.04)
    plt.legend()
    fig.set_figheight(4)
    fig.set_figwidth(5.5)
    
main()