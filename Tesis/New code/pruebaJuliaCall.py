import numpy as np
from juliacall import Main as jl
import time

jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/paths_EmptyRoom.jl")
jl.seval("using .pathsEmptyRoom")
dom=jl.EmptyRoom(0.4,0.6)
t0=time.time()
resp=jl.simulate_N_samples(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3,1)
print(time.time()-t0)
print("Ya compilado")
t0=time.time()
resp=jl.simulate_N_samples(dom,0.01,0.001,0.0,1.0,[0.5,0.5,0.2,0.2,0.2,0.8],jl.Inf,3,1000)
print(time.time()-t0)

def f(x):
    return x**2

print(jl.test_f(f,3))