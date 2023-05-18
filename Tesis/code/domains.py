import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import int64, float64    # import the types
from numba import jitclass,prange,njit
from numba.typed import Dict
import time
import mpltex

spec = [
    ('N', int64),               # a simple scalar field
    ('dim', int64),
    ('total_time',float64),
    ('pInf',float64),
    ('pSup',float64),
    ('pAnc',float64)
]


#@jitclass(spec)
class EmptyRoom():
    "Class representing an domain that is an empty room with one exit"
    def __init__(self,N,total_time,pInf,pSup):
        self.N=N#Number of agents
        self.dim=2*self.N #Total dimension 
        self.total_time=total_time #Terminal time of the domain
        self.pInf=pInf #Lower point of the door in the y axis
        self.pSup=pSup #Upper point of the door in the y axis
        self.pAnc=self.pSup-self.pInf #Width of thr door 

    def exited_domain(self,X0,X1):
        if X1[0]<0 or X1[0]>1 or X1[1]>1 or X1[1]<0:
            v=X1-X0
            tleft=-X0[0]/(v[0])
            tup=(1-X0[1])/(v[1])
            trig=(1-X0[0])/(v[0])
            tdow=-X0[1]/(v[1])
            ts=np.array([tleft,tup,trig,tdow])
            ts=np.where(ts>1,10.0,ts)
            ts=np.where(ts<0,10.0,ts)
            i=np.argmin(ts)

            if i==0:
                Xi=np.array([0.0,X0[1]+ts[i]*(v[1])])
                return Xi,'Neu',ts[i]
            elif i==1:
                Xi=np.array([X0[0]+ts[i]*(v[0]),1.0])
                return Xi,'Neu',ts[i]
            elif i==2:
                Xi=np.array([1.0,X0[1]+ts[i]*(v[1])])
                if Xi[1]>self.pInf and Xi[1]<self.pSup:
                    return Xi,'Dir',ts[i]
                else:
                    return Xi,'Neu',ts[i]
            elif i==3:
                Xi=np.array([X0[0]+ts[i]*(v[0]),0.0])
                return Xi,'Neu',ts[i]
        return X1,'Non',1.0
    
    def one_agent_brownian(self,sig,dt,Nmax,t0,X0,dirichlet_cut=False,neumann_cut=False):
        sqdt=np.sqrt(dt)
        X=np.zeros((Nmax,2))
        X[0]=X0
        xis=np.random.normal(loc=0,scale=1,size=(Nmax-1,2))
        dtf=1.0
        for i in range(1,Nmax):
            Xnew=X[i-1]+sqdt*sig*xis[i-1]
            Xnew,exit,dtf=self.exited_domain(X[i-1],Xnew)
            if exit=='Neu':
                if not neumann_cut:
                    Xn=Xnew+np.sqrt(dt*(1-dtf))*sig*np.random.normal(loc=0,scale=1,size=2)
                    Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    while e!='Non':
                        Xn=Xnew+np.sqrt(dt*(1-dtf))*sig*np.random.normal(loc=0,scale=1,size=2)
                        Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    X[i]=Xs
                    xis[i-1]=(X[i]-X[i-1])/(sqdt*sig)
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Dir':
                if not dirichlet_cut:
                    X[i]=Xnew
                    X[i+1:]=np.array([1.0,self.pInf+0.5*self.pAnc])
                    xis[i:]=np.zeros(2)
                    return X,xis,dtf
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Non':
                X[i]=Xnew
        return X,xis,dtf
    
    def simulate_one_controlled_diffusion(self,control,sig,dt,Nmax,t0,X0,dirichlet_cut=False,neumann_cut=False):
        sqdt=np.sqrt(dt)
        X=np.zeros((Nmax,2))
        X[0]=X0
        xis=np.random.normal(loc=0,scale=1,size=(Nmax-1,2))
        dtf=1.0
        t=t0
        for i in range(1,Nmax):
            mu=control(t,X[i-1])
            Xnew=X[i-1]+mu*dt+sqdt*sig*xis[i-1]
            Xnew,exit,dtf=self.exited_domain(X[i-1],Xnew)
            if exit=='Neu':
                if not neumann_cut:
                    mu=control(t,Xnew)
                    Xn=Xnew+mu*dt+np.sqrt(dt*(1-dtf))*sig*np.random.normal(loc=0,scale=1,size=2)
                    Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    while e!='Non':
                        Xn=Xnew+mu*dt+np.sqrt(dt*(1-dtf))*sig*np.random.normal(loc=0,scale=1,size=2)
                        Xs,e,dtf2=self.exited_domain(Xnew,Xn)
                    X[i]=Xs
                    xis[i-1]=(X[i]-X[i-1])/(sqdt*sig)
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Dir':
                if not dirichlet_cut:
                    X[i]=Xnew
                    X[i+1:]=np.array([1.0,self.pInf+0.5*self.pAnc])
                    xis[i:]=np.zeros(2)
                    return X,xis,dtf
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Non':
                X[i]=Xnew
            t+=dt
        return X,xis,dtf
    
    def simulate_difussion_N_agents_path(self,sig,dt,N_max,t0,X0):
        if t0>self.total_time-dt:
            t0-=2*dt
        tsim=int((self.total_time-t0)/dt)
        Nsim=min(N_max,tsim)+2
        Xc,xic,dtf=self.one_agent_brownian(sig,dt,Nsim,t0,X0[:2],True,False)
        X=np.zeros((Xc.shape[0],self.dim))
        Xis=np.zeros((Xc.shape[0]-1,self.dim))
        X[:,0:2]=Xc
        Xis[:,0:2]=xic

        for i in prange(1,self.N):
            Xi,xi,dtfi=self.one_agent_brownian(sig,dt,Xc.shape[0],t0,X0[2*i:2*i+2],False,False)
            X[:,2*i:2*i+2]=Xi
            Xis[:,2*i:2*i+2]=xi
        return X,xi,dtf

    
    #@mpltex.web_decorator
    def plot_N_brownian_paths(self,sig,dt,Nmax,t0,X0,N,dirichlet_cut=False,neumann_cut=False):
        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((1.0, self.pInf),0 , self.pAnc, linewidth=5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.set_xlim((-0.2,1.2))
        ax.set_ylim((-0.2,1.2))
        #plt.grid()
        ax.axis('equal')
        for _ in range(N):
            X0=np.random.uniform(low=0,high=1,size=2)
            Xs,xis,dtf=self.one_agent_brownian(sig,dt,Nmax,t0,X0,dirichlet_cut,neumann_cut)
            plt.scatter([Xs[-1][0]],[Xs[-1][1]],color='g')
            plt.plot(Xs[:,0],Xs[:,1])
        plt.show()
        #fig.savefig('images/sapo.pdf')
        return 0
    
    def plot_controlled_diffusion(self,control,sig,dt,Nmax,t0,X0,N,dirichlet_cut=False,neumann_cut=False):
        fig, ax = plt.subplots(1)
        rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((1.0, self.pInf),0 , self.pAnc, linewidth=5, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)
        ax.set_xlim((-0.2,1.2))
        ax.set_ylim((-0.2,1.2))
        #plt.grid()
        ax.axis('equal')
        for _ in range(N):
            X0=np.random.uniform(low=0,high=1,size=2)
            Xs,xis,dtf=self.simulate_controlled_diffusion(control,sig,dt,Nmax,t0,X0,dirichlet_cut,neumann_cut)
            plt.scatter([Xs[-1][0]],[Xs[-1][1]],color='g')
            plt.plot(Xs[:,0],Xs[:,1])
        plt.show()
        #fig.savefig('images/sapo.pdf')
        return 0
    
    def plot_N_agent_sample_path(self,X):
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



    def interior_points_sample(self,num_sample):
        t=np.random.uniform(low=0,high=self.total_time,size=[num_sample,1])
        x=np.random.uniform(size=(num_sample,self.dim))
        return np.hstack((t,x))
    
    def dirichlet_sample(self,num_sample):
        t=np.random.uniform(low=0,high=self.total_time,size=[num_sample,1]) 
        x=np.stack((np.ones(num_sample),self.pInf+np.random.uniform(size=num_sample)*self.pAnc),axis=1)
        return np.hstack((t,x))
    
    def neumann_sample(self,num_sample):
        Ns=int(num_sample/4)
        iz=np.stack((np.zeros(Ns),np.random.uniform(size=Ns)),axis=1)
        niz=np.repeat([[-1.0,0.0]],Ns,0)
        up=np.stack((np.random.uniform(size=Ns),np.ones(Ns)),axis=1)
        nup=np.repeat([[0.0,1.0]],Ns,0)
        down=np.stack((np.random.uniform(size=Ns),np.zeros(Ns)),axis=1)
        ndown=np.repeat([[0.0,-1.0]],Ns,0)
        der=np.stack((np.ones(Ns),0.2+np.random.uniform(size=Ns)*0.8),axis=1)
        nder=np.repeat([[1.0,0.0]],Ns,0)
        x=np.concatenate((iz,up,down,der))
        t=np.random.uniform(low=0,high=self.total_time,size=[x.shape[0],1])
        return np.hstack((t,x)),np.concatenate((niz,nup,ndown,nder))
    
    def terminal_sample(self,num_sample):
        T=np.ones(shape=[num_sample,1])*self.total_time
        x=np.random.uniform(size=[num_sample,self.dim])
        return np.hstack((T,x))


#dom=EmptyRoom({"N":1,"total_time":1.0,"pInf":0.4,"pSup":0.6})
dom=EmptyRoom(5,1.0,0.4,0.6)
nu=0.05
lam=4
def control_posible(t,X):
    n=np.array([1.0-X[0],0.5-X[1]])
    return 2*np.sqrt(lam) *n/np.linalg.norm(n)

sig=np.sqrt(2*nu)
#dom.plot_controlled_diffusion(control_posible,sig,0.001,2000,0.0,[0.1,0.5],1,dirichlet_cut=True,neumann_cut=False)


start = time.perf_counter()
X,a,t=dom.one_agent_brownian(sig,0.001,20,0.0,np.array([0.1,0.5]),False,False)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))

start = time.perf_counter()
X0=np.random.uniform(low=0,high=1,size=(dom.N*2))
X0=np.array([0.1,0.5,0.95,0.5,0.95,0.6,0.01,0.9,0.5,0.5])
X,a,t=dom.simulate_difussion_N_agents_path(sig,0.001,2000,0.0,X0)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))


dom.plot_N_agent_sample_path(X)