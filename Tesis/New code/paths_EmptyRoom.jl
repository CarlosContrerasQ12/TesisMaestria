module pathsEmptyRoom
export EmptyRoom, simulate_one_path_Nagents, simulate_N_samples,test_f,simulate_N_samples_threaded,simulate_one_controlled_path_Nagents
#using FLoops

struct EmptyRoom
    pInf::Float64
    pSup::Float64
    pAnc::Float64
    EmptyRoom(pInf,pSup)=new(pInf,pSup,pSup-pInf)
end

function exited_domain(dom::EmptyRoom,X0,X1)
    if X1[1]<0.0 || X1[1]>1.0 || X1[2]>1.0 || X1[2]<0.0
        v=X1-X0
        tleft=-X0[1]/(v[1])
        tup=(1-X0[2])/(v[2])
        trig=(1-X0[1])/(v[1])
        tdow=-X0[2]/(v[2])
        ts=[tleft,tup,trig,tdow]
        ts=@. ifelse(ts > 1.0, 10.0, ts)
        ts=@. ifelse(ts < 0.0, 10.0, ts)
        i=argmin(ts)
        if i==1
            Xi=[0.0,X0[2]+ts[i]*(v[2])]
            return Xi,"Neu",ts[i]     
        elseif i==2
            Xi=[X0[1]+ts[i]*(v[1]),1.0]
            return Xi,"Neu",ts[i]
        elseif i==3
            Xi=[1.0,X0[2]+ts[i]*(v[2])]
            if Xi[2]>dom.pInf && Xi[2]<dom.pSup
                return Xi,"Dir",ts[i]
            else
                return Xi,"Neu",ts[i]
            end
        elseif i==4
            Xi=[X0[1]+ts[i]*(v[1]),0.0]
            return Xi,"Neu",ts[i]
        end
    end
    return X1,"Non",1.0
end

function one_agent_brownian(dom::EmptyRoom,sig,dt,path_length,X0,dirichlet_cut=false,neumann_cut=false)
    sqdt=sqrt(dt)
    X=zeros((path_length,2))
    X[1,:]=X0
    xis=randn((path_length-1,2))
    dtf=1.0
    for i in 2:path_length
        Xn=X[i-1,:]+sqdt*sig*xis[i-1,:]
        Xnew,exit,dtf=exited_domain(dom,X[i-1,:],Xn)
        if exit=="Neu"
            if !neumann_cut
                Xn=Xnew+sqrt(dt*(1-dtf))*sig*randn(2)
                Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
                while e!="Non"
                    Xn=Xnew+sqrt(dt*(1-dtf))*sig*randn(2)
                    Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
                end
                X[i,:]=Xs
                xis[i-1,:]=(X[i,:]-X[i-1,:])/(sqdt*sig)
            else
                X[i,:]=Xnew
                return X[begin:i],xis[begin:i-1],dtf
            end
        elseif exit=="Dir"
            if !dirichlet_cut
                X[i,:]=Xnew
                X[i+1:end,:]=vcat(repeat([[1.0,dom.pInf+0.5*dom.pAnc]],size(X[i+1:end,:])[1])...)
                xis[i:end,:].=0.0
                return X,xis,dtf
            else
                X[i,:]=Xnew
                return X[begin:i,:],xis[begin:i-1,:],dtf
            end
        elseif exit=="Non"
            X[i,:]=Xnew
        end
    end
    return X,xis,dtf
end

function simulate_one_path_Nagents(dom::EmptyRoom,sigma,dt,t0,total_time,X0,Nmax,Nagents,dirichlet_cut=true,neumann_cut=false)
    if t0>total_time-dt
        t0-=2*dt
    end
    tsim=Int(floor((total_time-t0)/dt))
    Nsim=Int(min(Nmax,tsim))#+2
    Xc,xic,dtf=one_agent_brownian(dom,sigma,dt,Nsim,X0[1:2],dirichlet_cut,neumann_cut)
    X=zeros((size(Xc)[1],Nagents*2))
    Xis=zeros((size(Xc)[1]-1,Nagents*2))
    X[:,1:2]=Xc
    Xis[:,1:2]=xic

    for i in 1:(Nagents-1)
        Xi,xi,dtfi=one_agent_brownian(dom,sigma,dt,size(Xc)[1],X0[2*i+1:2*i+2],false,false)
        X[:,2*i+1:2*i+2]=Xi
        Xis[:,2*i+1:2*i+2]=xi
    end
    t=collect(LinRange(t0,(size(Xc)[1]-1)*dt,size(Xc)[1]))
    t[end]=t[end-1]+dt*dtf
    #return np.hstack((t.reshape((Xc.shape[0],1)),X)),Xis
    return t,X,Xis
end

function simulate_N_brownian_samples(dom::EmptyRoom,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples)
    samples=Array{Any}(undef,n_samples)
    for i in 1:n_samples
        resp=simulate_one_path_Nagents(dom,sigma,dt,t0,total_time,X0,Nmax,Nagents)
        samples[i]=resp
    end
    return samples
end

function simulate_N_brownian_samples_threaded(dom::EmptyRoom,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples)
    samples=Array{Any}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        resp=simulate_one_path_Nagents(dom,sigma,dt,t0,total_time,X0,Nmax,Nagents)
        samples[i]=resp
    end
    return samples
end

function next_position(dom,dt,sqdt,sig,Xi,Xf)
    Xnew,exit,dtf=exited_domain(dom,Xi,Xf)
    if exit=="Non"
        return Xnew,exit,dtf
    elseif exit=="Neu"
        Xn=Xnew+sqrt(dt*(1-dtf))*sig*randn(2)
        Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
        while e!="Non"
            Xn=Xnew+sqrt(dt*(1-dtf))*sig*randn(2)
            Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
        end
        return Xs,exit,1.0
    elseif exit=="Dir"
        #Xs=[1.0,dom.pInf+0.5*dom.pAnc]
        Xs=[1.0,Xnew[2]]
        return Xs,exit,dtf
    end
end

function simulate_one_controlled_path_Nagents(dom::EmptyRoom,drift,sigma,dt,t0,total_time,X0,Nmax,Nagents,dirichlet_cut=true,neumann_cut=false)
    sqdt=sqrt(dt)
    if t0>total_time-dt
        t0-=2*dt
    end
    tsim=Int(floor((total_time-t0)/dt))
    Nsim=Int(min(Nmax,tsim))
    X=zeros((2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn((2*Nagents,Nsim-1))
    
    states=repeat([false],Nagents)
    t=zeros(Nsim)
    t[1]=t0
    for i in 2:Nsim
        Xn=X[:,i-1]+drift(t[i-1],X[:,i-1])*dt+sqdt*sigma*xis[:,i-1]
        #Xnew,exit,dtf=exited_domain(dom,X[1:2,i-1],Xn[1:2])
        Xnew,exit,dtf=next_position(dom,dt,sqdt,sigma,X[1:2,i-1],Xn[1:2])
        if exit=="Non" || exit=="Neu"
            t[i]=t[i-1]+dt
            X[1:2,i]=Xnew
            if exit=="Neu"
                xis[1:2,i-1]=(Xnew-X[1:2,i-1])/(sqdt*sigma)
            end
            for j in 2:Nagents
                if !states[j]
                    Xnewj,exitj,dtfj=next_position(dom,dt,sqdt,sigma,X[2*j-1:2*j,i-1],Xn[2*j-1:2*j])
                    if exitj=="Non"
                        X[2*j-1:2*j,i]=Xnewj
                    elseif exitj=="Neu"
                        X[2*j-1:2*j,i]=Xnewj
                        xis[2*j-1:2*j,i-1]=(Xnewj-X[2*j-1:2*j,i-1])/(sqdt*sigma)
                    elseif exitj=="Dir"
                        states[j]=true
                        X[2*j-1:2*j,i:end]=hcat(repeat([Xnewj],size(X[2*j-1:2*j,i:end])[2])...)
                    end
                end
            end
        elseif exit=="Dir"
            if dirichlet_cut
                states[1]=true
                t[i]=t[i-1]+dt*dtf
                X[1:2,i]=Xnew
                Xn2=X[:,i-1]+drift(t[i-1],X[:,i-1])*dt*dtf+sqdt*sqrt(dtf)*sigma*xis[:,i-1]
                for j in 2:Nagents
                    if !states[j]
                        Xnewj,exitj,dtfj=next_position(dom,dt,sqdt,sigma,X[2*j-1:2*j,i-1],Xn2[2*j-1:2*j])
                        if exitj=="Non"
                            X[2*j-1:2*j,i]=Xnewj
                        elseif exitj=="Neu"
                            X[2*j-1:2*j,i]=Xnewj
                            xis[2*j-1:2*j,i-1]=(Xnewj-X[2*j-1:2*j,i-1])/(sqdt*sigma)
                        elseif exitj=="Dir"
                            states[j]=true
                            X[2*j-1:2*j,i:end]=hcat(repeat([Xnewj],size(X[2*j-1:2*j,i:end])[2])...)
                        end
                    end
                end
                return t[begin:i],X[:,begin:i],xis[:,begin:i],states

            else 
                t[i]=t[i-1]+dt
                if states[1]==false
                    X[1:2,i:end]=hcat(repeat([Xnew],size(X[1:2,i:end])[2])...)
                    states[1]=true
                end
                for j in 2:Nagents
                    if !states[j]
                        Xnewj,exitj,dtfj=next_position(dom,dt,sqdt,sigma,X[2*j-1:2*j,i-1],Xn[2*j-1:2*j])
                        if exitj=="Non"
                            X[2*j-1:2*j,i]=Xnewj
                        elseif exitj=="Neu"
                            X[2*j-1:2*j,i]=Xnewj
                            xis[2*j-1:2*j,i-1]=(Xnewj-X[2*j-1:2*j,i-1])/(sqdt*sigma)
                        elseif exitj=="Dir"
                            states[j]=true
                            X[2*j-1:2*j,i:end]=hcat(repeat([Xnewj],size(X[2*j-1:2*j,i:end])[2])...)
                        end
                    end
                end
            end

        end
    end
    return t,X,xis,states
end

function test_f(f,a)
    return f(a)
end

end

using .pathsEmptyRoom

#dom=EmptyRoom(0.4,0.6);
#X,xis,dtf=one_agent_brownian(dom,0.01,0.01,10,[0.5,0.5],true,false);
#t,X,Xis=simulate_one_path_Nagents(dom,0.01,0.01,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],10000,3);
#N=3
#X0=ones(2*N)*0.5
#X0=[0.5,0.5,0.2,0.2,0.2,0.8]
#@time resp=simulate_N_samples(dom,0.01,0.001,0.0,1.0,X0,Inf,N,1000);

"""function drift(t,X)
    return zeros(size(X))
end

dom=EmptyRoom(0.4,0.6)
N=3
X0=[0.5,0.5,0.2,0.2,0.2,0.8]
t,X,xis,states=simulate_one_controlled_path_Nagents(dom,drift,0.01,0.01,0.0,1.0,X0,Inf,N)"""