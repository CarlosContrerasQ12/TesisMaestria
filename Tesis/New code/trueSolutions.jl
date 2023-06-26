module trueSolutions
export true_solution_LQR,terminal_cost_log,F
type=Float32
using PythonCall

function terminal_cost_log_un(sample)
    t,X,xis,states=sample
    #print(t)
    return log.((1.0.+sum(X[:,end].*X[:,end]))./2)
end

function terminal_cost_log(samples)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    Threads.@threads for i in 1:Nsamp
        toto[i]=terminal_cost_log_un(samples[i])
    end
    return toto
end

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

function F(x,samp)
    return 0.0
end

function calculate_running_cost_term(samp,F)
    t,X,xis,states=samp
    termF=0.0
    for i in 1:(size(X)[2]-1)
        termF+=F(X[:,i],states)*(t[i+1]-t[i])
    end
    return termF
end

function true_solution_LQR(samples,F,terminal_cost_values,lam,nu)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    Threads.@threads for i in 1:Nsamp
        toto[i]=calculate_running_cost_term(samples[i],F)
    end
    new=exp.(-(lam/nu).*(toto.+terminal_cost_values))
    mean=sum(new)/Nsamp
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