"""
The main file to solve parabolic partial differential equations (PDEs).
"""
import sys
sys.path.append('./pdeSolver')

from domains import *
from equations import *
from torch_solvers import *
import torch
import numpy as np
import time



def l_schedule(epoch):
    return 10.0/(epoch+1)

if __name__ == '__main__':
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("Comienza a corre a las ", current_time)
    dom_config={"total_time":1.0,"pInf":0.4,"pSup":0.6}
    dom=EmptyRoom(dom_config)
    eqn_config={"N":1,"nu":0.05,"lam":4.0}
    eqn=HJB_LQR_Equation_2D(dom,eqn_config)
    solver_params={"initial_lr":0.01,
               "initial_loss_weigths":[2.0,1.0,3.0,1.0],
               "net_size":{"width":3,"depth":50,"weigth_norm":False},
               "logging_interval":200,
               "dtype":torch.float32,
               "N_samples_per_batch":512,
               "sample_every":10}

    sol=DGM_solver(eqn,solver_params)
    sol.lr_schedule=l_schedule

    namesol='LQR_DGM_N=1_solution'
    nametrain='LQR_DGM_N=1_training'
    namemodel='LQR_DGM_N=1_model'

    if len(sys.argv)>1:
        namesol=namesol+'_'+sys.argv[1]
        nametrain=nametrain+'_'+sys.argv[1]
        namemodel=namemodel+'_'+sys.argv[1]

    sol.save_sol(namesol+'.sol')
    training=sol.train(1000)
    np.save(nametrain+'.npy',training)
    sol.save_model("./models/"+namemodel+".pth")
