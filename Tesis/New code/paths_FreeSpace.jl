module pathsFreeSpace

export simulate_N_brownian_samples,simulate_N_controlled_samples

const type=Float32


function simulate_one_brownian_path_Nagents(sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut=true,neumann_cut=false)
    dt=(total_time-t0)/(Ntdis-1)
    sqdt=sqrt(dt)
    if t0>total_time-dt
        t0-=2*dt
        dt=(total_time-t0)/(Ntdis-1)
    end
    #tsim=Int(floor((total_time-t0)/dt))
    Nsim=Int(min(Nmax,Ntdis))

    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    
    states=repeat([false],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i]=X[:,i-1]+sqdt*sigma*xis[:,i-1]
    end
    return t,X,xis,states
end

function simulate_one_controlled_path_Nagents(drift,sigma,Ntdis,t0,total_time,X0,Nmax,Nagents,dirichlet_cut=true,neumann_cut=false)
    dt=(total_time-t0)/(Ntdis-1)
    sqdt=sqrt(dt)
    if t0>total_time-dt
        t0-=2*dt
        dt=(total_time-t0)/(Ntdis-1)
    end
    #tsim=Int(floor((total_time-t0)/dt))
    Nsim=Int(min(Nmax,Ntdis))


    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    
    states=repeat([false],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i]=X[:,i-1]+drift(t[i-1],X[:,i-1])*dt+sqdt*sigma*xis[:,i-1]
    end
    return t,X,xis,states
end

function simulate_N_brownian_samples(sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Bool}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        resp=simulate_one_brownian_path_Nagents(sigma,dt,t0,total_time,X0,Nmax,Nagents)
        samples[i]=resp
    end
    return samples
end

function simulate_N_controlled_samples(drift,sigma,dt,t0,total_time,X0,Nmax,Nagents,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Bool}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        resp=simulate_one_controlled_path_Nagents(drift,sigma,dt,t0,total_time,X0,Nmax,Nagents)
        samples[i]=resp
    end
    return samples
end


end

using .pathsFreeSpace