import numpy as np
import multiprocessing
from juliacall import Main as jl
import os
print(os.cpu_count())
jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/file.jl")
sa=jl.my_func(10000)
print(sa)

"""def f(p):
    return jl.my_func(p)
with multiprocessing.Pool(4) as pool:
    areas = pool.map(f,range(10000))
print(areas)
#"""
#