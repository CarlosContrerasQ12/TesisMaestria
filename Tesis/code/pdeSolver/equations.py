import numpy as np
import torch
from domains import *

from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

 

class HJB_LQR_Equation_Room():
    """
    A class representing an HJB equation with boundary conditions for the value function 
    of a process that is trying to get out of a room in 2D
    """
    def __init__(self,dom_config, eqn_config):
        eqn_config["Equation"]='HJB_LQR_Equation_2D'
        domType=globals()[dom_config["Domain"]]
        self.eqn_config=eqn_config
        self.domain=domType(dom_config) #Domain of the equation
        self.N=eqn_config["N"] #Number of agents
        self.dim=2*self.N
        self.nu=eqn_config["nu"] #Parameter controlling the volatility of the process
        self.lam=eqn_config["lam"] #Paramenter controlling the control strentgh
        self.sig=np.sqrt(2*self.nu)
    
    def h_n(self,x):
        """Neumann boundary condition"""
        return 0.0
    
    def h_d(self,x):
        """Dirichlet boundary condition"""
        return 0.0
    
    def g_Tf(self,x):
        """Terminal condition"""
        return 0.0
    
    def F(self,x):
        """Panic function"""
        return 5.0

    def f(self,x,V,nV_x2):
        """
        Non linear part on the equation. It does not appear in the diffusion part. 
        nV_x2 is the squared norm of the gradient
        """
        return -self.lam*(nV_x2)+self.F(x)
        
    def Nv(self, x,V, V_t,nV_x2,V_xx):
        """Satisfaction operator, should be zero for a true solution """
        return V_t + self.nu*self.dim*V_xx +self.f(x,V,nV_x2)
    
    def control(self,t,V_x):
        return -np.sqrt(self.lam)*V_x
    
    def interior_point_sample(self,num_sample):
        return self.domain.interior_points_sample(num_sample,self.N)
    
    def dirichlet_sample(self,num_sample,i):
        return self.domain.dirichlet_sample(num_sample,self.N,i)
    
    def neumann_sample(self,num_sample,i):
        return self.domain.neumann_sample(num_sample,self.N,i)
    
    def terminal_sample(self,num_sample):
        return self.domain.terminal_sample(num_sample,self.N)
    
    def interior_diffusion_sample(self,dt,N_max,t0,X0):
        return self.domain.simulate_difussion_N_agents_path(self.sig,dt,N_max,self.N,t0,X0)
    
    def simulate_N_interior_diffusion(self,dt,N_max,t0,X0,Ndifussions):
        return self.domain.simulate_N_difussions_Nagents(self.sig,dt,N_max,self.N,t0,X0,Ndifussions)

class HJB_LQR_Unbounded():
    """
    A class representing an HJB equation in free space for the value function 
    of a LQR process
    """
    def __init__(self,dom_config, eqn_config):
        eqn_config["Equation"]='HJB_LQR_Unbounded'
        domType=globals()[dom_config["Domain"]]
        self.eqn_config=eqn_config
        self.domain=FreeSpace(dom_config)
        self.N=eqn_config["N"] #Number of agents
        self.dim_per_agent=eqn_config["dim_per_agent"]
        self.dim=self.dim_per_agent*self.N
        self.nu=eqn_config["nu"] #Parameter controlling the volatility of the process
        self.lam=eqn_config["lam"] #Paramenter controlling the control strentgh
        self.sig=np.sqrt(2*self.nu)
        self.final_point=torch.Tensor(np.array([0.2,0.5,0.8,0.5]))

    def true_solution(self,t0,X0,Nsim,dt):
        Nlong=int((self.domain.total_time-t0)/dt)
        dw_sample=torch.tensor(np.random.normal(size=[Nsim,self.dim,Nlong])*np.sqrt(dt)).requires_grad_(False)
        x_sample = torch.zeros([Nsim, self.dim,Nlong]).requires_grad_(False)
        F=0.0
        x_sample[:, :, 0] = torch.ones([Nsim, self.dim],requires_grad=False) * X0
        for i in range(Nlong-1):
            x_sample[:, :, i + 1] = x_sample[:, :, i] + self.sig * dw_sample[:, :, i]
            F+=self.F(t0+i*dt,x_sample[:,:,i])*dt
        term1=-(self.lam/self.nu)*self.g_tf(x_sample[:,:,-1])
        term2=-(self.lam/self.nu)*F
        return -(self.nu/self.lam)*torch.log(torch.mean(torch.exp(term1+term2)))
    
    def f_tf(self, t, x, y, z):
        return -self.lam * torch.sum(torch.square(z), 1, keepdims=True) +self.F(t,x)

    def g_tf(self, x):
        #return torch.sum(torch.square(x-torch.ones((x.shape[0],self.dim))*self.final_point),axis=1, keepdims=True)
        return torch.log((1 + torch.sum(torch.square(x), 1, keepdims=True)) / 2)

    def F(self,t,x):
        #print(torch.square(x[:,0]-x[:,2]).shape)
        #print(torch.square(x[:,1]-x[:,3]).shape)
        #dist=torch.square(x[:,0]-x[:,2])+torch.square(x[:,1]-x[:,3])+1.0
        #return 2*torch.reciprocal(dist)
        return 0.0

    def simulate_N_interior_diffusion(self,dt,Ndis,Ndifussions,X0):
        return self.domain.diffusion_brownian_sample(Ndifussions,self.dim,dt,Ndis,self.sig,X0)

    def simulate_controlled_trajectory(self,t0,X0,control=None):
        X=np.zeros(shape=(self.Ndis,self.dim))
        X[0]=X0
        t=t0
        if control!=None:
            for i in range(1,self.Ndis):
                X[i]=X[i-1]+self.lam*np.sqrt(2)*control(t,X[i-1])*self.dt+\
                    self.sigma*np.random.normal(size=(self.dim))*self.sqrt_dt
                t+=self.dt
        else:
            for i in range(1,self.Ndis):
                X[i]=X[i-1]+self.sigma*np.random.normal(size=(self.dim))*self.sqrt_dt
                t+=self.dt
        return X
    
    def plot_trajectory(self,X):
        fig, ax = plt.subplots(1)
        ax.axis('equal')
        ax.set_xlim((-0.2,1.2))
        ax.set_ylim((-0.2,1.2))
        
        for i in range(self.N):
            plt.plot(X[:,2*i],X[:,2*i+1])
            plt.scatter([X[:,2*i][0]],[X[:,2*i+1][0]],color='g',label='P0'+str(i+1))
        plt.legend()
        plt.show()

    
"""
eqn_config={"N":1,"nu":0.05,"lam":4.0}
dom=EmptyRoom(1.0,0.4,0.6)
eqn=HJB_LQR_Equation_2D(dom,eqn_config)


x=torch.tensor(np.array([[3.0,4.0],[5.0,6.0]]),requires_grad=True)
V=x[:,0]**2+x[:,1]**2
V_x=torch.autograd.grad(V,x,grad_outputs=torch.ones_like(V),retain_graph=True,create_graph=True,only_inputs=True)[0]
V_xx=torch.autograd.grad(V_x,x,grad_outputs=torch.ones_like(V_x),retain_graph=True,create_graph=True,only_inputs=True)[0]
print(V_x)
nV2=torch.sum(V_x*V_x,axis=1)

print(eqn.f(x,V,nV2))
"""