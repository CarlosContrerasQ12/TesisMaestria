module trueSolutions
export true_solution_LQR
type=Float32

function terminal_cost(samp,h_d_i,g_tf_i)
    t,X,xis,states=samp
    termgh=0.0
    for i in 1:Int(size(X)[1]/2)
        if states[i]!=-1
            termgh+=h_d_i(X[2*i-1:2*i,end],i)
        else
            termgh+=g_tf_i(X[2*i-1:2*i,end],i)
        end
    end
end

function terminal_cost_torch(samp,h_d_i,g_tf_i)
    t,X,xis,states=samp
    termgh=0.0
    for i in 1:Int(size(X)[1]/2)
        if states[i]!=-1
            termgh+=h_d_i(X[2*i-1:2*i,end],i)
        else
            termgh+=g_tf_i(X[2*i-1:2*i,end],i)
        end
    end
end

function terminal_cost_batched_list(samples,h_d_i,g_tf_i)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    for i in 1:Nsamp
        toto[i]=terminal_cost(samples[i],h_d_i,g_tf_i)
    end
    return toto
end

function terminal_cost_batched_torch(samples,h_d_i,g_tf_i)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    for i in 1:Nsamp
        toto[i]=terminal_cost(samples[i],h_d_i,g_tf_i)
    end
    return toto
end

function calculate_term(samples,F,h_d_i,g_tf_i,lam,nu)
    t,X,xis,states=samp
    termF=0.0
    for i in 1:(size(X)[2]-1)
        if 
        termF+=F(X[:,i],states)*(t[i+1]-t[i])
    end

    return exp(-(lam/nu)*(termF+termgh))

    
end

function true_solution_LQR(samples,F,h_d_i,g_tf_i,lam,nu)
    Nsamp=length(samples)
    toto = zeros(type, Nsamp)
    for i in 1:Nsamp
        toto[i]+=calculate_term(samples[i],F,h_d_i,g_tf_i,lam,nu)
    end
    mean=sum(toto)/Nsamp
    return -(nu/lam)*log(mean)
end


end

using .trueSolutions

"""include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_FreeSpace.jl")

function F(x)
    return 0.0
end

function g_tf(x)
    return log((1+sum(x.*x))/2)
end

function h_d(x)
    return 0.0
end
samp=simulate_N_brownian_samples(sqrt(2),0.001,0.0,1.0,zeros(100),Inf,50,10)

sol=true_solution_LQR(samp,F,h_d,g_tf,1.0,1.0)"""