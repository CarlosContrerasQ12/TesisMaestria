from julia import Julia
#Julia(compiled_modules=False,runtime='/home/carlos/julia-1.9.1/bin/julia')
jl=Julia(sysimage="/home/carlos/Documentos/Trabajo de grado/Tesis/New code/sys.so")
#from julia import Main as jl
#jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/allJuliaCode.jl")
jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/trueSolutions.jl")
import numpy as np
import matplotlib.pyplot as plt
from domains import Domain,FreeSpace
import time

import torch

types=torch.float32
torch.set_default_dtype(types)

class Equation():
    """
    An abstract class representing an equation to be solved
    """
    def true_solution(self,x):
        """True solution of the equation at a point"""
        raise("true_solution not implemented yet")
    def h_n(self,x):
        """Neuman boundary condition"""
        raise("h_n not implemented yet")
    
    def h_d(self,x):
        """Dirichlet boundary condition"""
        raise("h_n not implemented yet")
    
    def g_Tf(self,x):
        """Terminal condition"""
        raise("g_Tf not implemented yet")
    
    def f(self,x,u,u_x):
        """Non linear part of the equation"""
        raise("f not implemented yet")
    
    def Nv(self, x,V, V_t,nV_x2,V_xx):
        """Satisfaction operator, should be zero for a true solution """
        raise("Nv not implemented yet")
    
    def interior_points_sample(self,numsample):
        """Sample points inside domain of equation """
        raise("interior_points_sample not implemented yet")

    def dirichlet_sample(self,num_sample,i):
        """Sample points on dirichlet boundary"""
        raise("dirichlet_sample not implemented yet")
    
    def neumann_sample(self,num_sample,i):
        """Sample points on neumann boundary"""
        raise("neumann_sample not implemented yet")
    
    def terminal_sample(self,num_sample):
        """Sample terminal points in domain of equation """
        raise("terminal_sample not implemented yet")
    
    def simulate_brownian_diffusion_paths(self):
        """Simulate a brownian path inside the domain"""
        raise("simulate_brownian_diffusion not implemented yet")
    
    def simulate_controlled_diffusion_paths(self):
        """Simulate a controlled diffusion path inside the domain"""
        raise("simulate_brownian_diffusion not implemented yet")


class HJB_LQR_Equation(Equation):
    """
    A class representing an HJB equation with boundary conditions for the value function 
    of a process that is trying to get out of a room in 2D
    """
    def __init__(self, dom:Domain , eqn_config):
        eqn_config["Equation"]='HJB_LQR_Equation'
        self.eqn_config=eqn_config
        self.spatial_domain=dom
        self.N=eqn_config["N"] #Number of agents
        self.terminal_time=eqn_config["terminal_time"] #Terminal time of the equation
        self.dim=2*self.N #total dimension of equation
        self.nu=eqn_config["nu"] #Parameter controlling the volatility of the process
        self.lam=eqn_config["lam"] #Paramenter controlling the control strentgh
        self.sig=np.sqrt(2*self.nu)
        self.desired_final=np.ones(self.dim)*0.5
        self.terminal_cost_torch=torch.vmap(self.terminal_cost)

    def true_solution(self,t0,X0,Ntdis,Nsim,Nbatch):
        Nb=int(Nsim/Nbatch)
        resp=0.0
        def sam_gen():
            return self.simulate_brownian_diffusion_paths(Ntdis,t0,X0,np.inf,1)[0]
        for _ in range(Nb+1):
            resp+=jl.true_solution_LQR(sam_gen,1000,self.F,self.terminal_cost,self.lam,self.nu)
            jl.GC.gc()
        return resp/(Nb+1)
    
    
    def h_n(self,x):
        """Neumann boundary condition"""
        return 0.0
    
    def h_d_i(self,x,i):
        """Dirichlet boundary condition"""
        return 0.0
    
    def g_Tf_i(self,x,i):
        """Terminal condition"""
        #i=i-1
        #dist=x[2*i,2*i+1]-np.array([0.5,0.5])
        #return np.sum(dist*dist)
        #return np.log((1.0+np.sum(x*x))/2)
        return 0
    
    def terminal_cost(self,sample):
        t,X,xis,states=sample
        #X=torch.tensor(X)#.requires_grad_(False)
        #return torch.log((1.0+torch.sum(torch.square(X[:,-1])))/2)
        return np.log((1.0+np.sum(X[:,-1]*X[:,-1]))/2)
        termgh=0.0
        for i in range(X.shape[0]/2):
            if states[i]!=-1:
                termgh+=self.h_d_i(X[2*i-1:2*i,-1],i)
            else:
                termgh+=self.g_tf_i(X[2*i-1:2*i,-1],i)
        return termgh
    
    def F(self,x,states):
        """
        Panic function
        x is an unbatched numpy array
        """
        return 0.0
    
    def F_torch(self,x,states):
        """ 
        Panic function
        x is an batched torch tensor
        """
        return torch.zeros(x.shape[0])

    def f(self,t,x,y,z):
        """
        Non linear part on the equation. It does not appear in the diffusion part.
        x,y,z are unbatched numpy arrays
        """
        return -self.lam*(np.sum(z*z))+self.F(x)
    
    def f_torch(self,t,x,y,z,states):
        """
        Non linear part on the equation. It does not appear in the diffusion part.
        x,y,z are batched torch tensors
        """
        return -self.lam*(torch.sum(torch.square(z),dim=1))+self.F_torch(x,states)
        
    def Nv(self, x,V, V_t,nV_x2,V_xx):
        """Satisfaction operator, should be zero for a true solution """
        return V_t + self.nu*self.dim*V_xx +self.f(x,V,nV_x2)
    
    def control(self,t,z):
        return -np.sqrt(self.lam)*z
    
    def sample_initial_point(self):
        return self.spatial_domain.sample_initial_point(self.N)

    def simulate_brownian_diffusion_paths(self,Ntdis,t0,X0,N_max,n_samples):
        return self.spatial_domain.simulate_brownian_diffusion_paths(self.sig,Ntdis,t0,self.terminal_time,X0,N_max,self.N,n_samples)
    
    def simulate_controlled_diffusion_paths(self,drift,Ntdis,t0,X0,N_max,n_samples):
        return self.spatial_domain.simulate_controlled_diffusion_paths(drift,self.sig,Ntdis,t0,self.terminal_time,X0,N_max,self.N,n_samples)
    
    def modify_brownian_diffusion_paths(self,samples,Ntdis,t0,X0,N_max,n_samples):
        self.spatial_domain.modify_brownian_diffusion_paths(self,samples,self.sigma, Ntdis, t0, self.total_time, X0, N_max, n_samples)

    def modify_controlled_diffusion_paths(self,samples,drift,Ntdis,t0,X0,N_max,n_samples):
        self.spatial_domain.modify_brownian_diffusion_paths(self,samples,drift,self.sigma, Ntdis, t0, self.total_time, X0, N_max, n_samples)



"""begin=time.time()
dom.simulate_brownian_diffusion_paths(np.sqrt(2),1001,0.0,1.0,np.zeros(100),np.inf,50,1)
print(time.time()-begin)
begin=time.time()
samp=dom.simulate_brownian_diffusion_paths(np.sqrt(2),1001,0.0,1.0,np.zeros(100),np.inf,50,1000)
print(time.time()-begin)
begin=time.time()
dom.modify_brownian_diffusion_paths(samp,np.sqrt(2),1001,0.0,1.0,np.zeros(100),np.inf,1000)
print(time.time()-begin)

begin=time.time()
dom.modify_brownian_diffusion_paths(samp,np.sqrt(2),1001,0.0,1.0,np.zeros(100),np.inf,1000)
print(time.time()-begin)
print("aca")"""

dom=FreeSpace({"nada":0.2})

eqn_config={"N":50,"terminal_time":1.0,"nu":1.0,"lam":1.0}
eqn=HJB_LQR_Equation(dom,eqn_config)
print(eqn.true_solution(0.0,np.zeros(100),1001,10000,100))
