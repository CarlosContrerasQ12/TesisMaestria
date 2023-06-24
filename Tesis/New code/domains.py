import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from julia import Julia
#Julia(compiled_modules=False,runtime='/home/carlos/julia-1.9.1/bin/julia')
Julia(sysimage="/home/carlos/Documentos/Trabajo de grado/Tesis/New code/sys.so")
from julia import Main as jl

class Domain():
    """
    An abstract class containing all functions that should provide a 2D-domain.
    """
    def simulate_brownian_diffusion_path(self,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,n_samples,dirichlet_cut=False,neumann_cut=False):
        """
        This function should simulate a difussion inside the domain without drift.
        sigma: Constant volatility of the process
        dt: Time step for the simulation
        t0: Initial time of simulation
        total_time: Final time simulation
        X0: Initial point of simulation, must have shape 2*Nagents
        Nmax: Maximum numbers of time steps to be considered, could be inf
        Nagents: Number of agents that are inside the room
        n_samples: number of paths to be simulated
        dirichlet_cut: Determines if the simulation must stop if reached the dirichlet boundary
        neummann_cut: Determines if the simulation stops if reached the neumann boundary. If false the motion is reflected
        """
        raise("simulate diffusion path not implemented yet")
    def simulate_controlled_diffusion_path(self,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,n_samples,dirichlet_cut=False,neumann_cut=False):
        """
        This function should simulate a difussion inside the domain without drift.
        drift: A function representing the drift of the process
        sigma: Constant volatility of the process
        dt: Time step for the simulation
        t0: Initial time of simulation
        total_time: Final time of simulation
        X0: Initial point of simulation, must have shape 2*Nagents
        Nmax: Maximum numbers of time steps to be considered, could be inf
        Nagents: Number of agents that are inside the room
        dirichlet_cut: Determines if the simulation must stop if reached the dirichlet boundary
        neummann_cut: Determines if the simulation stops if reached the neumann boundary. If false the motion is reflected
        """
        raise("simulate_controlled_diffusion_not_implemented yet")
    
    def plot_sample_path(self,X):
        """
        Plots a siffusion in the space
        """
        raise("plot_sample_path not implemented yet")
    
class FreeSpace(Domain):
    """
    A class representing free space
    """
    def __init__(self,dom_config):
        jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_FreeSpace.jl")
        dom_config["Domain"]="FreeSpace"
        self.dom_config=dom_config
    
    def simulate_brownian_diffusion_path(self, sigma, Ntdis, t0, total_time, X0, Nmax, Nagents, n_samples):
        if Nmax==np.inf:
            resp=jl.simulate_N_brownian_samples(sigma,Ntdis,t0,total_time,X0,jl.Inf,Nagents,n_samples)
        else:
            resp=jl.simulate_N_brownian_samples(sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,n_samples)
        return resp
    
    def simulate_controlled_diffusion_path(self,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,n_samples):
        if Nmax==np.inf:
            resp=jl.simulate_N_controlled_samples(drift,sigma,Ntdis,t0,total_time,X0,jl.Inf,Nagents,n_samples)
        else:
            resp=jl.simulate_N_controlled_samples(drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,n_samples)
        return resp 
    
    def plot_sample_path(self,X):
        fig, ax = plt.subplots(1)
        ax.set_xlim((-0.2,1.2))
        ax.set_ylim((-0.2,1.2))
        #plt.grid()
        ax.axis('equal')
        for i in range(int(X.shape[0]/2)):
            plt.plot(X[2*i,:],X[2*i+1,:])
            plt.scatter([X[2*i,-1]],[X[2*i+1,-1]],color='g')
            plt.scatter([X[2*i,0]],[X[2*i+1,0]],color='g')
        return 0

class EmptyRoom(Domain):
    """A class representing a domain [0,1]x[0,1] that is an empty room with one exit door
    -N is the number of people inside the room
    -total_time is the end time of the simulation
    -pInf is the lower part of the door in the y axis
    -pSup is the Upper part of the door in the y axis
    """
    def __init__(self,dom_config):
        jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_EmptyRoom.jl")
        dom_config["Domain"]="EmptyRoom"
        self.dom_config=dom_config
        self.pInf=dom_config["pInf"] #Lower point of the door in the y axis
        self.pSup=dom_config["pSup"] #Upper point of the door in the y axis
        self.pAnc=self.pSup-self.pInf #Width of thr door 
        self.dom_jl=jl.EmptyRoom(self.pInf,self.pSup)
    
    def simulate_brownian_diffusion_path(self, sigma, Ntdis, t0, total_time, X0, Nmax, Nagents,dirichlet_cut, n_samples):
        #jl.GC.enable(False)
        if Nmax==np.inf:
            resp=jl.simulate_N_brownian_samples_sim(self.dom_jl,sigma,Ntdis,t0,total_time,X0,jl.Inf,Nagents,dirichlet_cut,n_samples)
        else:
            resp=jl.simulate_N_brownian_samples_sim(self.dom_jl,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut,n_samples)
        #jl.GC.enable(True)
        return resp
    
    def simulate_controlled_diffusion_path(self,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut,n_samples):
        #jl.GC.enable(False)
        if Nmax==np.inf:
            resp=jl.simulate_N_controlled_samples_sim(self.dom_jl,drift,sigma,Ntdis,t0,total_time,X0,jl.Inf,Nagents,dirichlet_cut,n_samples)
        else:
            resp=jl.simulate_N_controlled_samples_sim(self.dom_jl,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut,n_samples)
        #jl.GC.enable(True)
        return resp 

    def plot_sample_path(self,X):
        """
        Plot an interior sample for the interpolation PINN-BSDE algorithm
        """
        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((1.0, self.pInf),0 , self.pAnc, linewidth=5, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.set_xlim((-0.2,1.2))
        ax.set_ylim((-0.2,1.2))
        #plt.grid()
        ax.axis('equal')
        for i in range(int(X.shape[0]/2)):
            plt.plot(X[2*i,:],X[2*i+1,:])
            plt.scatter([X[2*i,-1]],[X[2*i+1,-1]],color='g')
            plt.scatter([X[2*i,0]],[X[2*i+1,0]],color='g')
        return 0

print("aca")
dom=EmptyRoom({"pInf":0.4,"pSup":0.6})
#dom=FreeSpace({"s":0.3})
N=3
sig=0.1
Ntdis=1001
n_samples=1000
X0=[0.95,0.5,0.2,0.2,0.2,0.8]

def drift(t,X):
    dir=np.array([1.0,0.5,1.0,0.5,1.0,0.5])-X
    return 10*dir/np.sqrt(np.sum(dir*dir))
    #return np.zeros(shape=X.shape)

"""print("Comienza a calcular")
start = time.time()
#resp=dom.simulate_controlled_diffusion_path(drift,sig,Ntdis,0.0,1.0,X0,np.inf,N,1)
resp=dom.simulate_brownian_diffusion_path(sig,Ntdis,0.0,1.0,X0,np.inf,N,False,1)
end = time.time()
print("Elapsed (after compilation) = {}s".format((end - start)))
dom.plot_sample_path(resp[0][1])
print(resp[0][3])
plt.show()


print("Comienza a calcular")
start = time.time()
#resp=dom.simulate_controlled_diffusion_path(drift,sig,Ntdis,0.0,1.0,X0,np.inf,N,2)
resp=dom.simulate_brownian_diffusion_path(sig,Ntdis,0.0,1.0,X0,np.inf,N,False,1000)
end = time.time()
print("Elapsed (after compilation) = {}s".format((end - start)))
dom.plot_sample_path(resp[0][1])
print(len(resp))
plt.show()"""


"""ssl=np.random.normal(size=(int(1.0/dt),2))*np.sqrt(dt)*sig
Xb_test=np.cumsum(ssl,axis=0)+np.array([0.5,0.5])
dom.plot_sample_path(Xb_test.T)
plt.show()
"""


