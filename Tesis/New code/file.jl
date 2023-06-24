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

@time generate_samples(1);
@time generate_samples(1000);