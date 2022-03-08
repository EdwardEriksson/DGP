# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:48:21 2021

@author: Edwar
"""

import numpy as np
import matplotlib.pyplot as plt 
        
def sample_posterior(mesh_size,layers,number_draws,betas,data,gamma):
    mesh_count = int(np.round(1/mesh_size)+1)
    
    xi = [np.random.randn(mesh_count) for i in range(layers)]
    prediction = prop(xi,layers,mesh_size)
    
    B = get_B(betas)
    A = np.sqrt(np.eye(layers)-B**2)
    gamma_fac = np.linalg.cholesky(np.linalg.inv(gamma))
    
    samples = []
    total_accepted = 0
    total_rejected = 0
    for draw in range(number_draws):
        #generate a new state
        proposal = A@np.array(xi)+B@np.random.randn(layers,mesh_count)
        #figure out if we should update
        proposal_prediction = prop(proposal,layers,mesh_size)
        alpha = min(1,np.exp(phi(prediction,data,gamma_fac,mesh_size)-phi(proposal_prediction,data,gamma_fac,mesh_size)))
        if np.random.random() < alpha:
            total_accepted +=1
            xi = proposal
            prediction = proposal_prediction
        else:
            total_rejected += 1
        if draw % 1000 == 0:
            print(draw)
        samples.append(prediction)
    #print(total_accepted/(total_accepted+total_rejected))
    prop(xi,layers,mesh_size)
    return samples
        
def prop(xi,layers,mesh_size,plot_all=False):
    mesh_count = int(np.round(1/mesh_size)+1)
    mesh = np.linspace(0,1,mesh_count)
    covar = compute_initial_k(mesh)
    u = np.linalg.cholesky(covar)@xi[0]
    if plot_all == True:
        mesh_count = int(np.round(1/mesh_size)+1)
        mesh = np.linspace(0,1,mesh_count)
        plt.scatter(mesh,u,label="0") 
    for i in range(1,layers):
        covar = compute_covar(u,mesh_size)
        u = np.linalg.cholesky(covar)@xi[i]
        if plot_all == True:
            #print(np.round(covar,2))
            plt.scatter(mesh,u,label=i)
    if plot_all == True:
        plt.legend()
    return u

def phi(prediction,data,gamma_fac,mesh_size):
    #precompute cholesky(inv(gamma))
    #use linear solver
    prediction = sample_to_func(prediction, mesh_size)
    relevant_preds = np.array([prediction(obs_loc) for obs_loc in data["obs_locs"]])
    return 0.5*np.linalg.norm(gamma_fac@(relevant_preds-data["obs_values"]))**2

def get_B(betas):
    beta_count = len(betas)
    B = np.zeros((beta_count,beta_count))
    for i in range(beta_count):
        B[i][i] = betas[i]
    return B

#computing covariance        
def sample_to_func(sample,mesh_size):
    def u(x):
        #print(x)
        #print(x/mesh_size)
        return sample[int(round(x/mesh_size))]
    return u
    
def rho_s(x):
    return np.exp(-x**2)

def F(x):
    return x**2 
    
def compute_covar(sample,mesh_size):
    eps = 10**(-12)
    u = sample_to_func(sample,mesh_size)
    disc_count = int(round(1/mesh_size)+1)
    #Note: code written for 1-d regression
    sigma = 1
    c = np.zeros((disc_count,disc_count))
    for i in range(disc_count):
        for j in range(disc_count):
            x = i*mesh_size
            x_p = j*mesh_size
            #study pre_fac
            pre_fac = sigma**2*2**0.5*(F(u(x))*F(u(x_p)))**0.25/(F(u(x))+F(u(x_p)))**0.5
            c[i,j] = pre_fac*rho_s((np.sqrt(2)*abs(x-x_p))/(F(u(x))+F(u(x_p)))**0.5)
    return c+eps*np.eye(disc_count)

def initial_k(x,y):
    eps = 10**(-12)
    return np.exp(-abs(x-y)**2/(2*0.2**2))+eps*np.isclose(x,y)

def compute_initial_k(points):
    N = len(points)
    covar = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            covar[i][j] = initial_k(points[i],points[j])
    return covar

def GP(mesh,data,noise_param):
    def GP_kernel(x,y):
        l = 0.1
        return np.exp(-(x-y)**2/(2*l))
    
    def K(A,B):
        covar = np.zeros((len(A),len(B)))
        for i in range(len(A)):
            for j in range(len(B)):
                covar[i][j] = GP_kernel(A[i],B[j])
        return covar
    X = data["obs_locs"]
    X_s = list(mesh[:])
    for x in X:
        X_s.remove(x)
    T = np.linalg.inv((K(X,X)+noise_param**2*np.eye(len(X))))
    y = data["obs_values"]
    post_mean = K(X_s,X)@T@y
    post_covar = K(X_s,X_s)-K(X_s,X)@T@K(X,X_s)
    post_sample = np.random.multivariate_normal(post_mean,post_covar)
    return X_s,post_sample
    
    
def rmse(pred,true):
    pred = np.array(pred)
    true = np.array(true)
    return np.sqrt(np.sum((pred-true)**2)/len(pred))


    
def sin_recip(x):
    offset = 0.03
    if type(x) == float:
        return np.sin(1/(x+offset))
    elif type(x) == list:
        return [np.sin(1/(t+offset)) for t in x]
    
def sin(x):
    if type(x) == float:
        return np.sin(2*np.pi*x)
    elif type(x) == list:
        return [np.sin(2*np.pi*t) for t in x]
        
def step(x):
    if type(x) == float:
        return 0.8*(-1)**(x<0.5)
    elif type(x) == list:
        return [0.8*(-1)**(t<0.5) for t in x]

def main():
    target_funcs = [sin,step,sin_recip] #sin_recip
            
    mesh_size = 0.01    
    mesh_count = int(np.round(1/mesh_size)+1)
    mesh = np.linspace(0,1,mesh_count)
    data_interval = 2
    noise_param = 0.1
    layers = 2
    betas = [0.07]*layers
    draws = 3
    burn_in = 1
    n_chains = 3
    fig,ax = plt.subplots(1,3)
    i = 0
    titles = ["Smooth Function","Step Function","Sine Reciprocal"]
    for target_func in target_funcs:
        data = {"obs_values":[target_func(i*mesh_size) for i in range(0,mesh_count,data_interval)],"obs_locs":[mesh_size*i for i in range(0,mesh_count,data_interval)]}
        gamma = noise_param*np.eye(len(data["obs_values"]))
        
        chains = []
        for n in range(n_chains):
            chain = sample_posterior(mesh_size, layers,draws,betas,data,gamma)[burn_in:]
            chains.append(chain)
        chains = np.array(chains)
        posterior_mean = np.mean(chains,axis=(0,1))
        print(posterior_mean.shape)
        
        ax[i].scatter(data["obs_locs"],data["obs_values"],label = "Data",s=8)
        X_s,GP_sample = GP(mesh,data,noise_param)
        ax[i].scatter(X_s,GP_sample,label = "GP",s=8,marker="D")
        GP_rmse = rmse(GP_sample,target_func(X_s))
        posterior_mean = sample_to_func(posterior_mean, mesh_size)
        ax[i].scatter(mesh,[posterior_mean(x) for x in mesh],label="DGP",s=8,marker="x")
        DGP_rmse = rmse([posterior_mean(x) for x in X_s],target_func(X_s))
        ax[i].set(xlabel="x",ylabel="y",title = titles[i],xlim=(0,1),ylim=(-1.2,1.2))
        print(GP_rmse,"GP")
        print(DGP_rmse,"DGP")
        i += 1
        print(GR(chains),"GR")
    plt.suptitle("DGP Posterior Samples")
    plt.tight_layout()
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels,loc = (0.79,0.15),fontsize ="small")

def GR(chains):
    N = chains.shape[1]
    chain_vars = np.var(chains,axis=1,ddof=1)
    avg_vars = np.mean(chain_vars,axis=0)
    chain_means = np.mean(chains,axis=1)
    B_n = np.var(chain_means,axis=0,ddof=1)
    sig_sqr = ((N-1)/N)*avg_vars+B_n
    R = np.sqrt(np.divide(sig_sqr,avg_vars))
    return np.mean(R)


main()