import numpy as np
from numba.pycc import CC

cc = CC('calculate_fast_paths')

@cc.export('exited_domain','(f8,f8,f8[:],f8[:])')
def exited_domain(pInf,pSup,X0,X1):
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
                if Xi[1]>pInf and Xi[1]<pSup:
                    return Xi,'Dir',ts[i]
                else:
                    return Xi,'Neu',ts[i]
            elif i==3:
                Xi=np.array([X0[0]+ts[i]*(v[0]),0.0])
                return Xi,'Neu',ts[i]
        return X1,'Non',1.0

@cc.export('one_agent_brownian','(f8,f8,f8,f8,f8,i4,f8,f8[:],b1,b1)')
def one_agent_brownian(pInf,pSup,pAnc,sig,dt,Nmax,t0,X0,dirichlet_cut=False,neumann_cut=False):
        """ 
        Simulates ones brownian motion in the domain with the following parameters:
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
        X=np.zeros((Nmax,2))
        X[0]=X0
        xis=np.random.normal(loc=0,scale=1,size=(Nmax-1,2))
        dtf=1.0
        for i in range(1,Nmax):
            Xnew=X[i-1]+sqdt*sig*xis[i-1]
            Xnew,exit,dtf=exited_domain(X[i-1],Xnew)
            if exit=='Neu':
                if not neumann_cut:
                    Xn=Xnew+np.sqrt(dt*(1-dtf))*sig*np.random.normal(loc=0,scale=1,size=2)
                    Xs,e,dtf2=exited_domain(Xnew,Xn)
                    while e!='Non':
                        Xn=Xnew+np.sqrt(dt*(1-dtf))*sig*np.random.normal(loc=0,scale=1,size=2)
                        Xs,e,dtf2=exited_domain(Xnew,Xn)
                    X[i]=Xs
                    xis[i-1]=(X[i]-X[i-1])/(sqdt*sig)
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Dir':
                if not dirichlet_cut:
                    X[i]=Xnew
                    X[i+1:]=np.array([1.0,pInf+0.5*pAnc])
                    xis[i:]=np.zeros(2)
                    return X,xis,dtf
                else:
                    X[i]=Xnew
                    return X[:i+1],xis[:i],dtf
            elif exit=='Non':
                X[i]=Xnew
        return X,xis,dtf

@cc.export('simulate_difussion_N_agents_path','(f8,f8,f8,i4,i4,f8,f8[:])')
def simulate_difussion_N_agents_path(total_time,sig,dt,N_max,Nagents,t0,X0):
        if t0>total_time-dt:
            t0-=2*dt
        tsim=int((total_time-t0)/dt)
        Nsim=min(N_max,tsim)+2
        Xc,xic,dtf=one_agent_brownian(sig,dt,Nsim,t0,X0[:2],True,False)
        X=np.zeros((Xc.shape[0],Nagents*2))
        Xis=np.zeros((Xc.shape[0]-1,Nagents*2))
        X[:,0:2]=Xc
        Xis[:,0:2]=xic

        for i in range(1,Nagents):
            Xi,xi,dtfi=one_agent_brownian(sig,dt,Xc.shape[0],t0,X0[2*i:2*i+2],False,False)
            X[:,2*i:2*i+2]=Xi
            Xis[:,2*i:2*i+2]=xi
        t=np.linspace(t0,(Xc.shape[0]-1)*dt,Xc.shape[0])
        t[-1]=t[-2]+dt*dtf
        return np.hstack((t.reshape((Xc.shape[0],1)),X)),Xis,dtf

if __name__ == "__main__":
    cc.compile()