from julia import Julia
Julia(compiled_modules=False,runtime='/home/carlos/julia-1.9.1/bin/julia')
from julia import Base
from julia import Main as jl
import time
import numpy as np

jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_EmptyRoom.jl")
jl.eval("using .pathsEmptyRoom")
dom=jl.EmptyRoom(0.4,0.6)
sa=jl.simulate_N_samples(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3,1)
sa2=jl.simulate_N_samples_threaded(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3,1)


print("Ya compilado")


t0=time.time()
sa=jl.simulate_N_samples(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3,1000)
print("Sin threads:",time.time()-t0)

t0=time.time()
sa=jl.simulate_N_samples_threaded(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3,1000)
print("Con threads:",time.time()-t0)

print(type(sa[0][2]))

def f(x):
    return np.sum(x)

jl.test_f(f,np.array([1.0,2.0,3.0]))