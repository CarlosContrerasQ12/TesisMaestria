function my_func(N)
    samples=Array{Any}(undef,N)
    Threads.@threads for i in 1:N
        samples[i]=sum(ones(100000)*i)
    end
    return samples
end

function my_func2(N)
    samples=Array{Any}(undef,N)
    for i in 1:N
        samples[i]=sum(ones(100000)*i)
    end
    return samples
end

#@time my_func(10000)