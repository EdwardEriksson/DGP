# -*- coding: utf-8 -*-
"""
Let's fit a Gaussian process without the explicit solution'
"""


import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import math

def target_func(x):
    omega = 1
    return np.sin(2*np.pi*omega*x)

def make_data(train_points,noise_param):
    return target_func(train_points)+np.random.normal(loc=0,scale=noise_param,size=len(train_points))

def kernel(x,y):
    eps = 10**(-12)
    return np.exp(-abs(x-y)**2/(2*0.2**2))+eps*(abs(x-y)<0.01)

def compute_prior_covariance(points):
    N = len(points)
    covar = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            covar[i][j] = kernel(points[i],points[j])
    return covar

def set_up_prior(test_points):
    mean = np.array([0]*len(test_points))
    covar = np.array(compute_prior_covariance(test_points))
    return mean,covar

def matching_squared_sum(u,y,mesh_size):
    total = 0
    for x in y.keys():
        total += (y[x]-u[int(x/mesh_size)])**2
    return total
        
def unnormalized_posterior(u,y,prior_mean,prior_covar,noise_param,mesh_size):
    prior_eval = multivariate_normal.pdf(u,mean = prior_mean,cov=prior_covar)
    conditional_eval = np.exp((-len(y.keys())/(2*noise_param**2))*matching_squared_sum(u, y, mesh_size))
    return prior_eval*conditional_eval

#def gaussian_kernel_sample(x):
 #   z = np.random.normal(loc = x,scale = 0.1)
  #  return z
  
def crank_nicholson(x,prior_covar):
    beta = 0.12
    return np.sqrt(1-beta**2)*x+beta*np.random.multivariate_normal([0]*len(x),prior_covar)

def rho(x,z,y,prior_mean,prior_covar,noise_param,mesh_size):
    #z is the new proposal, x is the current state, y is the data
    proposal_score = np.exp((1/(2*noise_param**2))*(matching_squared_sum(x, y,mesh_size)-matching_squared_sum(z, y, mesh_size)))
    return min(proposal_score,1)

def metropolis_hastings_sample(N,y,prior_mean,prior_covar,noise_param,mesh_size,mh_kernel):
    accepted = 0
    rejected = 0
    test_points = np.linspace(0,1,21)
    x = prior_mean
    samples = []
    i = 0
    while i < N:
        z = mh_kernel(x,prior_covar)
        if rho(x,z,y,prior_mean,prior_covar,noise_param,mesh_size) > np.random.random():
            x = z
            samples.append(x)
            accepted += 1
        else:
            samples.append(x)
            rejected += 1
        i += 1
    return samples

def gaussian_kernel(x,y):
    eps = 0 #10**(-11)
    return 0.5*np.exp(-(x-y)**2/(2*0.3**2))+eps*(abs(x-y)<0.01)

def analytic_fitting(data,test_points,noise_param):
    def K(A,B):
        kernel_matrix = list(np.zeros((len(A),len(B))))
        for i in range(len(A)):
            for j in range(len(B)):
                kernel_matrix[i][j] = kernel(A[i],B[j])
        return np.array(kernel_matrix)
    y = [data[key] for key in data.keys()]
    noise_matrix = noise_param**2*np.identity(6)
    X = np.linspace(0,1,6)
    return K(test_points,X)@np.linalg.inv((K(X,X)+noise_matrix))@y

def mse(x,y):
    return np.sum((x-y)**2)/len(x)
    
def GR(chains):
    N = chains.shape[1]
    chain_vars = np.var(chains,axis=1,ddof=1)
    avg_vars = np.mean(chain_vars,axis=0)
    chain_means = np.mean(chains,axis=1)
    B_n = np.var(chain_means,axis=0,ddof=1)
    sig_sqr = ((N-1)/N)*avg_vars+B_n
    R = np.sqrt(np.divide(sig_sqr,avg_vars))
    return np.mean(R)

def main():

    mesh_size = 0.05
    train_point_count = 6
    noise_param = 0.25
    train_points = np.linspace(0,1,train_point_count)
    y =  dict(zip(train_points,make_data(train_points, noise_param)))
    samples_per_chain = 2000
    burn_in = 1000
    n_chains = 10
        
    test_point_count = int(1/mesh_size+1)
    test_points = np.linspace(0,1,test_point_count)
    prior_mean, prior_covar = set_up_prior(test_points)
        
    chains = []
    for i in range(n_chains):
        chain=metropolis_hastings_sample(samples_per_chain,y,prior_mean,prior_covar,noise_param,mesh_size,crank_nicholson)[burn_in:]
        chains.append(chain)
    chains = np.array(chains)
        
    y_keys = y.keys()
    true_y = [target_func(x) for x in test_points]
    plt.scatter(test_points,true_y,label = "True function, y = sin(2\u03C0x)",marker="o",s=20)
    plt.scatter(y_keys,[y[key] for key in y_keys],label="Data",marker="x",s=20)
    
    #plt.scatter(test_points,prior_mean)
    #print(chains.shape)
    posterior_experimental_mean = np.mean(chains,axis=(0,1))
    #print(posterior_experimental_mean)
            
    #plt.scatter(test_points,posterior_samples[-1],label="Posterior sample")
    plt.scatter(test_points,posterior_experimental_mean,label="Posterior MH mean",marker="s",s=20)
        
    analytic_mean = analytic_fitting(y,test_points,noise_param)
    #print(np.linalg.norm(posterior_experimental_mean-analytic_mean))
    plt.scatter(test_points,analytic_mean,label="Analytic mean",marker="*",s=20)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Metropolis-Hastings Mean GP")
    
    print(GR(chains))
    #analytic_sample_errors = [mse(analytic_mean,posterior_sample) for posterior_sample in posterior_samples]
    
    #plt.scatter([i for i in range(len(analytic_sample_errors))],analytic_sample_errors)

main()

