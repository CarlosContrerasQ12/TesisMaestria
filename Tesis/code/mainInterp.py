import sys
sys.path.append('./pdeSolver')

import numpy as np
from domains import *
from equations import *
from torch_solvers import *
import torch

print("Probando interp")
dom_config={"total_time":1.0,"pInf":0.4,"pSup":0.6}
dom=EmptyRoom(dom_config)
eqn_config={"N":1,"nu":0.05,"lam":4.0}
eqn=HJB_LQR_Equation_2D(dom,eqn_config)

solver_params={"initial_lr":0.001,
            "initial_loss_weigths":[2.0,1.0,1.0,1.0],
            "net_size":{"width":30,"depth":3,"weigth_norm":False},
            "logging_interval":10,
            "dtype":torch.float32,
            "dt":0.01,
            "N_max":25,
            "N_samples_per_batch_interior":200,
            "N_samples_per_batch_boundary":40,
            "sample_every":1,
            "alpha":0.09,
            "adaptive_weigths":False}
sol=Interp_PINN_BSDE_solver(eqn,solver_params)



namesol='LQR_INTERP_N_=1_solution'
nametrain='LQR_INTERP_N=1_training'
namemodel='LQR_INTERP_N=1_model'

if len(sys.argv)>1:
    namesol=namesol+'_'+sys.argv[1]
    nametrain=nametrain+'_'+sys.argv[1]
    namemodel=namemodel+'_'+sys.argv[1]

sol.save_sol(namesol+'.sol')
training=sol.train(30000)
np.save(nametrain+'.npy',training)
sol.save_model("./models/"+namemodel+".pth")
