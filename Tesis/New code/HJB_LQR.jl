module trueSolutions
export true_solution_LQR,terminal_cost_samples,modify_terminal_cost_samples,F_samples,modify_F_samples
type=Float32
"""function h_d_i(x,i)
    return 0.0
end

function g_tf_i(x,i)
    return 0.0
end

function terminal_cost(samp)
    t,X,xis,states=samp
    termgh=0.0
    for i in 1:Int(size(X)[1]/2)
        if states[i]!=-1
            @views termgh+=h_d_i(X[2*i-1:2*i,end],i)
        else
            @views termgh+=g_tf_i(X[2*i-1:2*i,end],i)
        end
    end
    return termgh
end"""


function terminal_cost(sample)
    t,X,xis,states=sample
    #print(t)
    return log.((1.0.+sum(X[:,end].*X[:,end]))./2)
end

function terminal_cost_samples(samples)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    Threads.@threads for i in 1:Nsamp
        toto[i]=terminal_cost(samples[i])
    end
    return toto
end

function modify_terminal_cost_samples(terminal_costs,samples)
    Threads.@threads for i in 1:length(terminal_costs)
        terminal_costs[i]=terminal_cost(samples[i])
    end
end

function F_pos(x,states,i)
    return 0.0
end

function F_sample(sample)
    t,X,xis,states=sample
    F_tot=zeros(type,size(X)[2])
    for i in 1:size(X)[2]
        F_tot[i]=@views F_pos(X[:,i],states,i)
    end
    return F_tot
end

function F_samples(samples)
    Nsamp=length(samples)
    F_tot = Vector{Vector{type}}(undef,Nsamp)
    #F_tot = zeros(type,(Nsamp,size(samples[1][2])[2]))
    Threads.@threads for i in 1:Nsamp
        F_tot[i]=@views F_sample(samples[i])
    end
    return F_tot
end

function modify_F_samples(F_samples_values,samples)
    Threads.@threads for i in 1:length(F_samples_values)
        F_samples_values[i]=@views F_sample(samples[i])
    end
end


function calculate_running_cost_term(F_sample,t)
    termF=0.0
    for i in 1:(size(F_sample)[1]-1)
        termF+=F_sample[i]*(t[i+1]-t[i])
    end
    return termF
end

function true_solution_LQR(samples,F_values,terminal_cost_values,lam,nu)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    Threads.@threads for i in 1:Nsamp
        toto[i]=calculate_running_cost_term(F_values[i],samples[i][1])
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

@time sol=true_solution_LQR(get_samp,F,terminal_cost_values,1.0,1.0) """ 