import numpy as np
import matplotlib.pyplot as plt
from julia import Julia
jl=Julia(sysimage="/home/carlos/Documentos/Trabajo de grado/Tesis/New code/sys.so")
jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/file.jl")

samp=jl.generate_samples(1000)
print(jl.calculate_index(samp))
