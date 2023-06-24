from julia import Julia
#Julia(compiled_modules=False,runtime='/home/carlos/julia-1.9.1/bin/julia')
Julia(sysimage="/home/carlos/Documentos/Trabajo de grado/Tesis/New code/sys.so")
from julia import Main as jl
import time

jl.include("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/file.jl")

start=time.time()
samp=jl.generate_samples(1000)
end=time.time()
print("Elpased time: ",end-start)
