function my_func(N)
    samples=Array{Float64}(undef,N)
    Threads.@threads for i in 1:N
      samples[i]=sum(i for i in 1:10^4)
    end
    return samples
 end

 @time my_func(10000)