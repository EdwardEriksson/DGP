import numpy as np
#np.random.seed(3141592)
#np.random.seed(11111)
np.random.seed(1111111)
import matplotlib.pyplot as plt
from scipy.stats import norm
pdf = norm.pdf
cdf = norm.cdf
from scipy.optimize import fmin

def find_max(f,x_0):
    def f_n(x):
        return -f(x)
    maximum = fmin(f_n,x_0,disp=False)
    return maximum[0]

target_x = [0,0.06,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
target_y = [0.2,0.1,0.2,0.3,0.9,0.8,0.6,0.5,0.5,0.7,0.0]
pol = np.polyfit(target_x,target_y,deg=10)

def f(x):
    return np.polyval(pol,x)
    #f_val = []
    #if not isinstance(x, np.ndarray):
    #    x = np.array([x])
    #for t in x:
    #    if 0 <= t and t <= 1:
    #        f_val.append(np.polyval(pol,t))
    #    else:
    #        f_val.append(0)
    #f_val = np.array(f_val)
    #return f_val

def K(A,B):
    covar = np.zeros((len(A),len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            covar[i][j] = kernel(A[i],B[j])
    return covar
    
def kernel(x,y):
    l = 0.1
    return np.exp(-(x-y)**2/(2*l**2))


def get_post(data):
    X = data["obs_locs"]
    y = data["obs_vals"]
    epsI = 10**(-12)*np.eye(len(X))
    def f_mean(x):
        x = np.array([x])
        return float(K(x,X)@np.linalg.inv(K(X,X)+epsI)@y)
    def f_var(x):
        x = np.array([x])
        var = float(K(x,x)-K(x,X)@np.linalg.inv(K(X,X)+epsI)@K(X,x))
        return var
    return f_mean,f_var
    
def get_expected_improvement_func(f_mean,f_var,f_max):
    def expected_improvement(x):
        delta = f_mean(x)-f_max
        if f_var(x) <= 0:
            return 0
        else:
            sigma = np.sqrt(f_var(x))
        return (f_mean(x)-f_max)*cdf((f_mean(x)-f_max)/sigma)+sigma*pdf((f_mean(x)-f_max)/sigma)
        #return max(0,delta)+sigma*pdf(delta/sigma)-abs(delta)*cdf(delta/sigma)
    return expected_improvement

def BayesianOptimization(mesh,f,iterations):
    data = {"obs_locs":np.array([0]),"obs_vals":np.array([f(0)])}
    f_max = max(data["obs_vals"])
    f_mean, f_var = get_post(data)
    expected_improvement = get_expected_improvement_func(f_mean, f_var, f_max)
    fig, ax = plt.subplots(nrows=1,ncols=2)
    
    fig.set_figheight(4)
    fig.set_figwidth(5.5)
    fig.suptitle("Sensitivity to Length Scale")
    titles = ["Intermediate Length Scale","Short Length Scale"]
    for r in range(2):
        if r==1:
            global kernel
            def kernel(x,y):
                l = 0.025
                return np.exp(-(x-y)**2/(2*l**2))
        data = {"obs_locs":np.array([0]),"obs_vals":np.array([f(0)])}
        for i in range(iterations):
            if i == iterations-1:
                
    
                ax[r].plot(mesh,f(mesh),label="f")
                ax[r].scatter(data["obs_locs"],data["obs_vals"],label="Evaluations")
                ax[r].plot(mesh,[expected_improvement(x) for x in mesh],label="Expected improvement")
                ax[r].plot(mesh,[f_mean(x) for x in mesh],label="Surrogate mean")
                ax[r].set(xlabel="x",ylabel="y",title=titles[r],xlim = (0,1),ylim=(-1,1.2))
                
                for j, txt in enumerate([k for k in range(1,iterations+1)]):
                    ax[r].annotate(txt, (data["obs_locs"][j], data["obs_vals"][j]))
    
            
            new_x = find_max(expected_improvement,np.random.random())
            if new_x < 0 or new_x > 1:
                print("Error. Left allowed region")
                return
            f_new = f(new_x)
            data["obs_locs"] = np.append(data["obs_locs"],new_x)
            data["obs_vals"] = np.append(data["obs_vals"],f_new)
            f_max = max(f_max,f_new)
            f_mean, f_var = get_post(data)
            expected_improvement = get_expected_improvement_func(f_mean, f_var, f_max)
            
        #ax[1].scatter(data["obs_locs"],data["obs_vals"],label="Evaluations")
        #ax[1].plot(mesh,f(mesh),label="f")
        #ax[1].set(xlabel="x",ylabel="y",title="After Extrapolation Changes",xlim=(0,1),ylim=(-1,1.2))
        #ax[1].plot(mesh,[expected_improvement(x) for x in mesh],label="Expected improvement")
        #for i, txt in enumerate([i for i in range(1,iterations+2)]):
        #    ax[1].annotate(txt, (data["obs_locs"][i], data["obs_vals"][i]))
        #ax[1].plot(mesh,[f_mean(x) for x in mesh],label="Surrogate mean")
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels,loc = (0.2,0.15),fontsize ="small")
        fig.tight_layout()
    return data

mesh_count = 1001
mesh = np.linspace(0,1,mesh_count)
data = BayesianOptimization(mesh,f,5)

print(data["obs_locs"][list(data["obs_vals"]).index(max(data["obs_vals"]))],max(data["obs_vals"]))
#plt.scatter(data["obs_locs"],data["obs_vals"])
    

    
    
