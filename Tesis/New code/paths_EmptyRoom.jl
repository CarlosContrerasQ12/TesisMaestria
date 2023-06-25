module pathsEmptyRoom


export EmptyRoom, simulate_N_brownian_samples_sim,simulate_N_controlled_samples_sim


const type=Float32
struct EmptyRoom
    pInf::type
    pSup::type
    pAnc::type
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

function next_position(dom,dt,sqdt,sig,Xi,Xf)
    Xnew,exit,dtf=exited_domain(dom,Xi,Xf)
    if exit=="Non"
        return Xnew,exit,dtf
    elseif exit=="Neu"
        Xn=Xnew+sqrt(dt*(1.0-dtf))*sig*randn(type,2)
        Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
        while e!="Non"
            Xn=Xnew+sqrt(dt*(1.0-dtf))*sig*randn(type,2)
            Xs,e,dtf2=exited_domain(dom,Xnew,Xn)
        end
        return Xs,exit,1.0
    elseif exit=="Dir"
        Xs=[1.0,Xnew[2]]
        return Xs,exit,dtf
    end
end

function fill_array!(X,Xnew,i,j)
    #X[2*j-1:2*j,i:end]=hcat(repeat([Xnew],size(X[2*j-1:2*j,i:end])[2])...)
    X[2*j-1,i:end].=Xnew[1]
    X[2*j,i:end].=Xnew[2]
end

function update_states!(dom,dt,sqdt,sigma,i,Nagents,states,X,Xn,xis)
    for j in 1:Nagents
        if states[j]==-1
            Xnewj,exitj,dtfj=next_position(dom,dt,sqdt,sigma,X[2*j-1:2*j,i-1],Xn[2*j-1:2*j])
            if exitj=="Non"
                X[2*j-1:2*j,i]=Xnewj
            elseif exitj=="Neu"
                X[2*j-1:2*j,i]=Xnewj
                xis[2*j-1:2*j,i-1]=(Xnewj-X[2*j-1:2*j,i-1])/(sqdt*sigma)
            elseif exitj=="Dir"
                states[j]=i
                @views fill_array!(X,Xnewj,i,j)
            end
        end
    end
end

function simulate_one_controlled_path_Nagents(dom::EmptyRoom,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut=[])
    dt=(total_time-t0)/(Ntdis-1)
    sqdt=sqrt(dt)
    if t0>total_time-dt
        t0-=2*dt
        dt=(total_time-t0)/(Ntdis-1)
    end
    Nsim=Int(min(Nmax,Ntdis))

    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    states=repeat([-1],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        Xn=X[:,i-1]+drift(t[i-1],X[:,i-1])*dt+sqdt*sigma*xis[:,i-1]
        update_states!(dom,dt,sqdt,sigma,i,Nagents,states,X,Xn,xis)
        if !isempty(dirichlet_cut)
            if all(states[dirichlet_cut].!=-1)
                return t[begin:i],X[:,begin:i],xis[:,begin:i],states
            end
        end
    end
    return t,X,xis,states
end

function simulate_one_brownian_path_Nagents(dom::EmptyRoom,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut=[])
    dt=(total_time-t0)/(Ntdis-1)
    sqdt=sqrt(dt)
    if t0>total_time-dt
        t0-=2*dt
        dt=(total_time-t0)/(Ntdis-1)
    end
    Nsim=Int(min(Nmax,Ntdis))

    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    states=repeat([-1],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        Xn=X[:,i-1]+sqdt*sigma*xis[:,i-1]
        update_states!(dom,dt,sqdt,sigma,i,Nagents,states,X,Xn,xis)
        if !isempty(dirichlet_cut)
            if all(states[dirichlet_cut].!=-1)
                return t[begin:i],X[:,begin:i],xis[:,begin:i],states
            end
        end
    end
    return t,X,xis,states
end

function simulate_N_brownian_samples_sim(dom::EmptyRoom,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Int}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        resp=simulate_one_brownian_path_Nagents(dom,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut)
        samples[i]=resp
    end
    return samples
end

function simulate_N_controlled_samples_sim(dom::EmptyRoom,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Int}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        resp=simulate_one_controlled_path_Nagents(dom,drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut)
        samples[i]=resp
    end
    return samples
end

end

using .pathsEmptyRoom

"""using PyCall
import PyPlot; const plt = PyPlot
pygui(true)
function plot_samp(X)
    fig,ax=plt.subplots(1)
    ax.set_xlim((-0.2,1.2))
    ax.set_ylim((-0.2,1.2))
    plt.hlines([0,1],0,1)
    plt.vlines([0,1],0,1)
    plt.vlines([1],0.4,0.6,color="r")
    for i in 1:Int(size(X)[1]/2)    
        plt.plot(X[2*i-1,:],X[2*i,:])
    end
    return 0
end

function drift(t,X)
    vec=[1.0,0.5,1.0,0.5,1.0,0.5]-X
    vec[1:2]=[0.0,0.0]
    return 10*vec./sqrt(sum(vec.*vec))
end

dom=EmptyRoom(0.4,0.6);
N=3
sig=0.5
Ntdis=1001
n_samples=1000
X0=[0.2,0.5,0.2,0.2,0.2,0.8]
#@time samp=simulate_one_brownian_path_Nagents(dom,sig,Ntdis,0.0,1.0,X0,Inf,N,false);
@time samp=simulate_N_controlled_samples_sim(dom,drift,sig,Ntdis,0.0,1.0,X0,Inf,N,[],1000);
X=samp[2][2]
plot_samp(X)
display(samp[2][2])"""