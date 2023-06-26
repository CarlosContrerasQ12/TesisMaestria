import numpy as np
from numba import njit, prange
import time

dtype2=np.float32

@njit
def simulate_one_brownian_path_Nagents(sigma,Nsim,dt,sqdt,t0,X0,Nagents):
    X=np.zeros((2*Nagents,Nsim),dtype=dtype2)
    X[:,0]=X0
    xis=np.random.randn(2*Nagents,Nsim-1)
    states=np.ones(Nagents)*-1
    t=np.zeros(Nsim,dtype=dtype2)
    for i in range(1,Nsim):
        t[i]=t[i-1]+dt
        X[:,i]=X[:,i-1]+sqdt*sigma*xis[:,i-1]
    return t,X,xis,states

@njit(parallel=True)
def simulate_N_brownian_samples(sigma,Nsim,dt,sqdt,t0,X0,Nagents,n_samples):
    t=np.zeros((n_samples,Nsim),dtype=dtype2)
    X=np.zeros((n_samples,2*Nagents,Nsim),dtype=dtype2)
    xis=np.random.randn(n_samples,2*Nagents,Nsim-1)
    states=np.ones((n_samples,Nagents))*-1
    for i in prange(n_samples):
        samples=simulate_one_brownian_path_Nagents(sigma,Nsim,dt,sqdt,t0,X0,Nagents)
        t[i]=samples[0]
        X[i]=samples[1]
        xis[i]=samples[2]
        states[i]=samples[3]
    return samples


ts=time.time()
simulate_N_brownian_samples(np.sqrt(2),1001,0.001,np.sqrt(0.001),0.0,np.zeros(100),50,1000)
print("Tiempo",time.time()-ts)
"""function simulate_one_brownian_path_Nagents(sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    states=repeat([-1],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i].=@views X[:,i-1].+sqdt.*sigma.*xis[:,i-1]
    end
    return t,X,xis,states
end

function simulate_one_controlled_path_Nagents(drift,sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    states=repeat([-1],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i]=@views X[:,i-1].+drift(t[i-1],X[:,i-1]).*dt.+sqdt.*sigma.*xis[:,i-1]
    end
    return t,X,xis,states
end


function simulate_N_brownian_samples(sigma,Nsim,dt,sqdt,t0,X0,Nagents,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Int}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        samples[i]=simulate_one_brownian_path_Nagents(sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    end
    return samples
end

function simulate_N_controlled_samples(drift,sigma,Nsim,dt,sqdt,t0,X0,Nagents,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Int}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        samples[i]=simulate_one_controlled_path_Nagents(drift,sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    end
    return samples
end"""