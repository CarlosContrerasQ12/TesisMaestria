import numpy as np
import torch

from domains import EmptyRoom

class HJB_LQR_Equation_2D():
    """
    A class representing an HJB equation with boundary conditions for the value function 
    of a process that is trying to get out of a room in 2D
    """
    def __init__(self,domain, eqn_config):
        eqn_config["Equation"]='HJB_LQR_Equation_2D'
        self.eqn_config=eqn_config
        self.domain=domain #Domain of the equation
        self.N=eqn_config["N"] #Number of agents
        self.dim=2*self.N
        self.nu=eqn_config["nu"] #Parameter controlling the volatility of the process
        self.lam=eqn_config["lam"] #Paramenter controlling the control strentgh
    
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
        return 2.0

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