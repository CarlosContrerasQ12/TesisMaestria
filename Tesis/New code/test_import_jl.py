import numpy as np
import time
import matplotlib.pyplot as plt
from domains import EmptyRoom
from julia import Julia
#Julia(compiled_modules=False,runtime='/home/carlos/julia-1.9.1/bin/julia')
jl=Julia(sysimage="/home/carlos/Documentos/Trabajo de grado/Tesis/New code/sys.so")
#from julia import Main as jl
jl.eval("print(2)")


print("aca")
dom=EmptyRoom({"pInf":0.4,"pSup":0.6},jl)
#dom=FreeSpace({"s":0.3})
N=3
sig=0.1
Ntdis=1001
n_samples=1000
X0=[0.95,0.5,0.2,0.2,0.2,0.8]

def drift(t,X):
    dir=np.array([1.0,0.5,1.0,0.5,1.0,0.5])-X
    return 10*dir/np.sqrt(np.sum(dir*dir))
    #return np.zeros(shape=X.shape)

print("Comienza a calcular")
start = time.time()
#resp=dom.simulate_controlled_diffusion_path(drift,sig,Ntdis,0.0,1.0,X0,np.inf,N,[],1)
resp=dom.simulate_brownian_diffusion_path(sig,Ntdis,0.0,1.0,X0,np.inf,N,[],1)
end = time.time()
print("Elapsed (after compilation) = {}s".format((end - start)))

start = time.time()
#resp=dom.simulate_controlled_diffusion_path(drift,sig,Ntdis,0.0,1.0,X0,np.inf,N,[],1)
resp=dom.simulate_brownian_diffusion_path(sig,Ntdis,0.0,1.0,X0,np.inf,N,[],1000)
end = time.time()
print("Elapsed (after compilation) = {}s".format((end - start)))


dom.plot_sample_path(resp[0][1])
print(resp[0][3])
plt.show()