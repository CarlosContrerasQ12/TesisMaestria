module trueSolutions
export true_solution_LQR
type=Float32

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
    termF=0.0
    """    new_states=zeros(size(states))

    for i in 1:Int(size(X)[1]/2)
        if states[i]==-1
            new_states[i]=Inf
        else
            new_states[i]=states[i]
        end
    end"""

    for i in 1:(size(X)[2]-1)
        termF+=F(X[:,i],states)*(t[i+1]-t[i])
    end
    termgh=type(terminal_cost_l(samp))
    return exp(-(lam/nu)*(termF+termgh))

    
end

function true_solution_LQR(gen_samp,Nsamp,F,terminal_cost,lam,nu)
    #Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    for i in 1:Nsamp
        toto[i]=calculate_term(gen_samp(),F,terminal_cost,lam,nu)
    end
    mean=sum(toto)/Nsamp
    return -(nu/lam)*log(mean)
end

end

using .trueSolutions

"""include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_FreeSpace.jl")

function F(x,states)
    return 0.0
end

function terminal_cost(sample)
    t,X,xis,states=sample
    return log((1.0+sum(X[:,end].*X[:,end]))/2)
end

function get_samp()
    return simulate_N_brownian_samples(sqrt(2),1001,0.001,sqrt(0.001),0.0,zeros(100),50,1)[1] 
end

@time sol=true_solution_LQR(get_samp,100000,F,terminal_cost,1.0,1.0) """ 