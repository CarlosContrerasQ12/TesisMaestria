
module pathsFreeSpace

export simulate_N_brownian_samples,simulate_N_controlled_samples,modify_N_brownian_samples,modify_N_controlled_samples

const type=Float32

function simulate_one_brownian_path_Nagents(sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    states=repeat([-1],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i].=@views X[:,i-1].+sqdt.*sigma.*xis[:,i-1]
    end
    return t,X,xis,states
end

function simulate_one_controlled_path_Nagents(drift,sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    X=zeros(type,(2*Nagents,Nsim))
    X[:,1]=X0
    xis=randn(type,(2*Nagents,Nsim-1))
    states=repeat([-1],Nagents)
    t=zeros(type,Nsim)
    t[1]=t0
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i]=@views X[:,i-1].+drift(t[i-1],X[:,i-1]).*dt.+sqdt.*sigma.*xis[:,i-1]
    end
    return t,X,xis,states
end


function simulate_N_brownian_samples(sigma,Nsim,dt,sqdt,t0,X0,Nagents,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Int}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        samples[i]=simulate_one_brownian_path_Nagents(sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    end
    return samples
end

function simulate_N_controlled_samples(drift,sigma,Nsim,dt,sqdt,t0,X0,Nagents,n_samples)
    samples=Array{Tuple{Vector{type},Matrix{type},Matrix{type},Vector{Int}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        samples[i]=simulate_one_controlled_path_Nagents(drift,sigma,Nsim,dt,sqdt,t0,X0,Nagents)
    end
    return samples
end

function modify_one_controlled_path_Nagents!(t,X,xis,states,drift,sigma,Nsim,dt,sqdt,t0,X0)
    t[1]=t0
    X[:,1].=X0
    xis.=randn.(type)
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i].=@views X[:,i-1].+drift(t[i-1],X[:,i-1]).*dt.+sqdt.*sigma.*xis[:,i-1]
    end
    return nothing
end

function modify_one_brownian_path_Nagents!(t,X,xis,states,sigma,Nsim,dt,sqdt,t0,X0)
    t[1]=t0
    X[:,1].=X0
    xis.=randn.(type)
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i].=@views X[:,i-1].+sqdt.*sigma.*xis[:,i-1]
    end
    return nothing
end

function modify_N_brownian_samples(samples,sigma,Nsim,dt,sqdt,t0,X0,n_samples)
    Threads.@threads for i in 1:n_samples
        modify_one_brownian_path_Nagents!(samples[i][1],samples[i][2],samples[i][3],samples[i][4],sigma,Nsim,dt,sqdt,t0,X0)
    end
    return samples
end

function modify_N_controlled_samples(samples,drift,sigma,Nsim,dt,sqdt,t0,X0,n_samples)
    Threads.@threads for i in 1:n_samples
        modify_one_controlled_path_Nagents!(samples[i][1],samples[i][2],samples[i][3],samples[i][4],drift,sigma,Nsim,dt,sqdt,t0,X0)
    end
    return samples
end

end

using .pathsFreeSpace

"""t0=0.0
X0=zeros(100)
sigma=sqrt(2)
total_time=1.0
Ntdis=1001
Nmax=Inf
Nagents=50
n_samples=1000
dt=(total_time-t0)/(Ntdis-1)
sqdt=dt
if t0>total_time-dt
    t0-=2*dt
    dt=(total_time-t0)/(Ntdis-1)
end
Nsim=Int(min(Nmax,Ntdis))

@time resp=simulate_N_brownian_samples(sigma,Nsim,dt,sqdt,t0,X0,Nagents,n_samples);

@time modify_N_brownian_samples(resp,sigma,Nsim,dt,sqdt,t0,X0,n_samples);"""