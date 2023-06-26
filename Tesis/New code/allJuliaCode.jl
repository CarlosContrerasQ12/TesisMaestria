const type=Float32

function terminal_cost(samp,h_d_i,g_tf_i)
    t,X,xis,states=samp
    termgh=0.0
    for i in 1:Int(size(X)[1]/2)
        if states[i]!=-1
            @views termgh+=h_d_i(X[2*i-1:2*i,end],i)
        else
            @views termgh+=g_tf_i(X[2*i-1:2*i,end],i)
        end
    end
end

function terminal_cost_batched_list(samples,h_d_i,g_tf_i)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    Threads.@threads for i in 1:Nsamp
        toto[i]=terminal_cost(samples[i],h_d_i,g_tf_i)
    end
    return toto
end

function terminal_cost_torch(samp,h_d_i,g_tf_i)
    t,X,xis,states=samp
    toto = zeros(type, Nsamp)
    Threads.@threads for j in 1:Int(size(X)[1])
        termgh=0.0
        for i in 1:Int(size(X)[2]/2)
            if states[j,i]!=-1
                termgh+=h_d_i(X[j,2*i-1:2*i,end],i)
            else
                termgh+=g_tf_i(X[j,2*i-1:2*i,end],i)
            end
        end
        toto[j]=termgh
    end
end

function calculate_term(samp,F,terminal_cost_l,lam,nu)
    t,X,xis,states=samp
    termF=zeros(type,size(X)[2]-1)
    """    new_states=zeros(size(states))

    for i in 1:Int(size(X)[1]/2)
        if states[i]==-1
            new_states[i]=Inf
        else
            new_states[i]=states[i]
        end
    end"""
"""
    for i in 1:(size(X)[2]-1)
        println("Entro aca",i)
        termF[i]=F(X[:,i],states)*(t[i+1]-t[i])
    end
 """   
    println("Yella")
    stermF=sum(termF)
    println("Yella2")
    termgh=terminal_cost_l(samp)
    #termgh=0.0

    return exp(-(lam/nu)*(stermF+termgh))

    
end

function true_solution_LQR(samples,F,terminal_cost,lam,nu)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    GC.enable(false)
    Threads.@threads for i in 1:Nsamp
        println(i)
        toto[i]+=calculate_term(samples[i],F,terminal_cost,lam,nu)
    end
    GC.enable(false)
    mean=sum(toto)/Nsamp
    return -(nu/lam)*log(mean)
end

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
    xis.=randn(type)
    for i in 2:Nsim
        t[i]=t[i-1]+dt
        X[:,i].=@views X[:,i-1].+drift(t[i-1],X[:,i-1]).*dt.+sqdt.*sigma.*xis[:,i-1]
    end
    return nothing
end

function modify_one_brownian_path_Nagents!(t,X,xis,states,sigma,Nsim,dt,sqdt,t0,X0)
    t[1]=t0
    X[:,1].=X0
    xis.=randn(type)
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