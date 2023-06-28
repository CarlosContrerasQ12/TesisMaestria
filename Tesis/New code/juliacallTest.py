
from juliacall import Main as jl
from domains import FreeSpace
from equations import HJB_LQR_Equation
import numpy as np

    
#jl.input("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/domains.py")
#jl.input("/home/carlos/Documentos/Trabajo de grado/Tesis/New code/trueSolutions.jl")


dom=FreeSpace({"nada":0.2})
eqn_config={"N":50,"terminal_time":1.0,"nu":1.0,"lam":1.0}
eqn=HJB_LQR_Equation(dom,eqn_config)

print(eqn.true_solution(0.0,np.zeros(100),1001,10000,100))

