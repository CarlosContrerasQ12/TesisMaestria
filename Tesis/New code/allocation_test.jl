const type=Float32
"""function simulate_path(path_length)
    xis=randn(type,(100,path_length-1))
    X=zeros(type,(100,path_length))
    sqdt=sqrt(0.001)
    sigma=sqrt(2)
    for i in 2:path_length
        X[:,i].=@views X[:,i-1].+sqdt.*sigma.*xis[:,i-1]
    end
    return X,xis
end

function simulate_N_paths(n_samples)
    samples=Array{Tuple{Matrix{type},Matrix{type}}}(undef,n_samples)
    Threads.@threads for i in 1:n_samples
        samples[i]=simulate_path(1000)
    end
    return samples
end

@time resp=simulate_N_paths(1000);

"""


function simulate_path!(X, xis, path_length)
    #xis .=randn.(type)
    sqdt=sqrt(0.001)
    sigma=sqrt(2)
    for i in 2:path_length
        X[:,i] .= @views X[:,i-1] .+ sqdt .* sigma .* xis[:,i-1]
    end
    return nothing
end

function simulate_N_paths!(n_samples,path_length)
    samples=[(zeros(type, (100, path_length)),randn(type, (100, path_length-1))) for i in 1:n_samples]
    Threads.@threads for i in 1:n_samples
        simulate_path!(samples[i][1], samples[i][2], 1000)
    end
    return samples
end

#resp=[(zeros(type, (100, 1001)),randn(type, (100, 1000))) for i in 1:1000]
function loco()
    for _ in 1:10
        @time resp=simulate_N_paths!(1000,1001);
        resp=nothing
    end
end
@time loco()