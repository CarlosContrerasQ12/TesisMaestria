function calculate_sample()
    sleep(0.001)
    return randn(1000),randn((6,1000)),randn((6,1000)),randn(6)
end

function generate_samples(N)
    samples=Array{Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64},Vector{Float64}}}(undef,N)
    Threads.@threads for i in 1:N
        samples[i]=calculate_sample()
    end
    return samples
end

function calculate_index(sample)
    result=zeros(length(sample))
    n_samp=length(sample)
    Threads.@threads for i in 1:n_samp
        result[i]=sum(sample[i][1])
    end
    return result
end