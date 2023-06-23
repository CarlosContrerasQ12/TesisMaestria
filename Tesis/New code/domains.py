import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from julia import Julia
Julia(compiled_modules=False,runtime='/home/carlos/julia-1.9.1/bin/julia')
from julia import Main as jl

jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_EmptyRoom.jl")

dtype=np.float32
np.random.normal2 = lambda *args,**kwargs: np.random.normal(*args, **kwargs).astype(dtype)
np.random.uniform2 = lambda *args,**kwargs: np.random.uniform(*args, **kwargs).astype(dtype)
np.ones2 = lambda *args,**kwargs: np.ones(*args, **kwargs).astype(dtype)
np.zeros2 = lambda *args,**kwargs: np.zeros(*args, **kwargs).astype(dtype)
np.array2 = lambda *args,**kwargs: np.array(*args, **kwargs).astype(dtype)

class Domain():
    """
    An abstract class containing all functions that should provide a 2D-domain.
    """
    def simulate_diffusion_path(self,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples,dirichlet_cut=False,neumann_cut=False):
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
    def simulate_controlled_diffusion_path(self,drift,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples,dirichlet_cut=False,neumann_cut=False):
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

class EmptyRoom(Domain):
    """A class representing a domain [0,1]x[0,1] that is an empty room with one exit door
    -N is the number of people inside the room
    -total_time is the end time of the simulation
    -pInf is the lower part of the door in the y axis
    -pSup is the Upper part of the door in the y axis
    """
    def __init__(self,dom_config):
        self.dom_config=dom_config
        self.total_time=dom_config["total_time"] #Terminal time of the domain
        self.dim=2
        self.pInf=dom_config["pInf"] #Lower point of the door in the y axis
        self.pSup=dom_config["pSup"] #Upper point of the door in the y axis
        self.pAnc=self.pSup-self.pInf #Width of thr door 
        self.dom_jl=jl.EmptyRoom(self.pInf,self.pSup)
    
    def simulate_diffusion_path(self, sigma, dt, t0, total_time, X0, Nmax, Nagents, n_samples, dirichlet_cut=False, neumann_cut=False):
        if Nmax==np.inf:
            resp=jl.simulate_N_samples_threaded(self.dom_jl,sigma,dt,t0,total_time,X0,jl.Inf,Nagents,n_samples)
        else:
            resp=jl.simulate_N_samples_threaded(self.dom_jl,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples)
        return resp
    
    def simulate_controlled_diffusion_path(self,drift,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples,dirichlet_cut=False,neumann_cut=False):
        if Nmax==np.inf:
            resp=jl.simulate_one_controlled_path_Nagents(self.dom_jl,drift,sigma,dt,t0,total_time,X0,jl.Inf,Nagents)
        else:
            resp=jl.simulate_one_controlled_path_Nagents(self.dom_jl,drift,sigma,dt,t0,total_time,X0,Nmax,Nagents)
        return resp 

    def exited_domain(self,X0,X1):
        """
        If the step from X0 to X1 exits the domain, it returns the following:
        -The point at which it touches the boundary
        -The type of boundary that it touched 'Neu' for the Neumann part 'Dir' for dirichlet part
        -The fraction of dt at which the movement touched it
        """
        if X1[0]<0 or X1[0]>1 or X1[1]>1 or X1[1]<0:
            v=X1-X0
            tleft=-X0[0]/(v[0])
            tup=(1-X0[1])/(v[1])
            trig=(1-X0[0])/(v[0])
            tdow=-X0[1]/(v[1])
            ts=np.array2([tleft,tup,trig,tdow])
            ts=np.where(ts>1,10.0,ts)
            ts=np.where(ts<0,10.0,ts)
            i=np.argmin(ts)

            if i==0:
                Xi=np.array2([0.0,X0[1]+ts[i]*(v[1])])
                return Xi,'Neu',ts[i]
            elif i==1:
                Xi=np.array2([X0[0]+ts[i]*(v[0]),1.0])
                return Xi,'Neu',ts[i]
            elif i==2:
                Xi=np.array2([1.0,X0[1]+ts[i]*(v[1])])
                if Xi[1]>self.pInf and Xi[1]<self.pSup:
                    return Xi,'Dir',ts[i]
                else:
                    return Xi,'Neu',ts[i]
            elif i==3:
                Xi=np.array2([X0[0]+ts[i]*(v[0]),0.0])
                return Xi,'Neu',ts[i]
        return X1,'Non',1.0
    
    def one_agent_brownian(self,sig,dt,Nmax,X0,dirichlet_cut=False,neumann_cut=False):
        """ 
        Simulates ones2 brownian motion in the domain with the following parameters:
        sig:: volatility of motion
        dt:: time step
        Nmax: maximum length of path
        t0: Starting time of simulation
        X0: Starting point
        dirichlet_cut: If true, the motion stops if it touches the dirihclet boundary.
        If false, it stays there until Nmax is reached
        neumann_cut: If true, the motion stops if it reaches the neumann boundary.
        If false, it simulates a reflected motion.

        """
        sqdt=np.sqrt(dt)
        X=np.zeros2((Nmax,2))
        X[0]=X0
        xis=np.random.normal2(loc=0,scale=1,size=(Nmax-1,2))
        dtf=1.0
        for i in range(1,Nmax):
            Xnew=X[i-1]+sqdt*sig*xis[i-1]
            Xnew,exit,dtf=self.exited_domain(X[i-1],Xnew)
            if exit=='Neu':
                if not neumann_cut:
                    Xn=Xnew+np.sqrt(dt*(1-dtf))*sig*np.random.normal2(loc=0,scale=1,size=2)
                    Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    while e!='Non':
                        Xn=Xnew+np.sqrt(dt*(1-dtf))*sig*np.random.normal2(loc=0,scale=1,size=2)
                        Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    X[i]=Xs
                    xis[i-1]=(X[i]-X[i-1])/(sqdt*sig)
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Dir':
                if not dirichlet_cut:
                    X[i]=Xnew
                    X[i+1:]=np.array2([1.0,self.pInf+0.5*self.pAnc])
                    xis[i:]=np.zeros2(2)
                    return X,xis,dtf
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Non':
                X[i]=Xnew
        return X,xis,dtf
    
    def simulate_one_controlled_diffusion(self,control,sig,dt,Nmax,t0,X0,dirichlet_cut=False,neumann_cut=False):
        sqdt=np.sqrt(dt)
        X=np.zeros2((Nmax,2))
        X[0]=X0
        xis=np.random.normal2(loc=0,scale=1,size=(Nmax-1,2))
        dtf=1.0
        t=t0
        for i in range(1,Nmax):
            mu=control(t,X[i-1])
            Xnew=X[i-1]+mu*dt+sqdt*sig*xis[i-1]
            Xnew,exit,dtf=self.exited_domain(X[i-1],Xnew)
            if exit=='Neu':
                if not neumann_cut:
                    mu=control(t,Xnew)
                    Xn=Xnew+mu*dt+np.sqrt(dt*(1-dtf))*sig*np.random.normal2(loc=0,scale=1,size=2)
                    Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    while e!='Non':
                        Xn=Xnew+mu*dt+np.sqrt(dt*(1-dtf))*sig*np.random.normal2(loc=0,scale=1,size=2)
                        Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    X[i]=Xs
                    xis[i-1]=(X[i]-X[i-1])/(sqdt*sig)
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Dir':
                if not dirichlet_cut:
                    X[i]=Xnew
                    X[i+1:]=np.array2([1.0,self.pInf+0.5*self.pAnc])
                    xis[i:]=np.zeros2(2)
                    return X,xis,dtf
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Non':
                X[i]=Xnew
            t+=dt
        return X,xis,dtf
    
    def simulate_difussion_N_agents_path(self,sig,dt,N_max,Nagents,t0,X0):
        if t0>self.total_time-dt:
            t0-=2*dt
        tsim=int((self.total_time-t0)/dt)
        Nsim=min(N_max,tsim)+2
        Xc,xic,dtf=self.one_agent_brownian(sig,dt,Nsim,X0[:2],True,False)
        X=np.zeros2((Xc.shape[0],Nagents*2))
        Xis=np.zeros2((Xc.shape[0]-1,Nagents*2))
        X[:,0:2]=Xc
        Xis[:,0:2]=xic

        for i in range(1,Nagents):
            Xi,xi,dtfi=self.one_agent_brownian(sig,dt,Xc.shape[0],X0[2*i:2*i+2],False,False)
            X[:,2*i:2*i+2]=Xi
            Xis[:,2*i:2*i+2]=xi
        t=np.linspace(t0,(Xc.shape[0]-1)*dt,Xc.shape[0])
        t[-1]=t[-2]+dt*dtf
        return np.hstack((t.reshape((Xc.shape[0],1)),X)),Xis
    
    def simulate_N_difussions_Nagents(self,sig,dt,N_max,Nagents,t0,X0,Ndifussions):
        samples=[]
        for i in range(Ndifussions):
            X,Xis=self.simulate_difussion_N_agents_path(sig,dt,N_max,Nagents,t0[i],X0[i])
            #resp=np.empty((2,N_max+2,Nagents*2))
            resp=np.empty((2,X.shape[0],X.shape[1]))
            resp.fill(np.nan)
            resp[0,:]=X
            resp[1,:-1,1:]=Xis
            samples.append(resp)
        return samples

    def plot_N_agent_sample_path(self,X):
        """
        Plot an interior sample for the interpolation PINN-BSDE algorithm
        """
        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((1.0, self.pInf),0 , self.pAnc, linewidth=5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.set_xlim((-0.2,1.2))
        ax.set_ylim((-0.2,1.2))
        #plt.grid()
        ax.axis('equal')
        for i in range(int(X.shape[1]/2)):
            plt.plot(X[:,2*i],X[:,2*i+1])
            plt.scatter([X[-1,2*i]],[X[-1,2*i+1]],color='g')
        plt.show()
        return 0

dom=EmptyRoom({"total_time":1.0,"pInf":0.4,"pSup":0.6})
N=3
nu=0.01
sig=np.sqrt(2*nu)
sig=0.4
dt=0.001
start = time.time()
n_samples=1000

X0=np.ones((n_samples,2*N))*0.5
X0=[0.5,0.5,0.2,0.2,0.9,0.5]
#dom.simulate_N_difussions_Nagents(sig,0.001,np.inf,N,t0,X0,n_samples)
#resp=dom.simulate_diffusion_path(sig,dt,0.0,1.0,X0,np.inf,3,n_samples)

def drift(t,X):
    return np.zeros(shape=X.shape)

t0=np.zeros(n_samples)
start = time.time()
resp=dom.simulate_controlled_diffusion_path(drift,sig,dt,0.0,1.0,X0,np.inf,3,1)
#print(resp[0])
end = time.time()
print("Elapsed (after compilation) = {}s".format((end - start)))

dom.plot_N_agent_sample_path(resp[1].T)