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

def l_schedule(epoch):
    return 1.0/(10*(epoch+1))

if __name__ == '__main__':
    dom=EmptyRoom(1.0,0.4,0.6)
    eqn_config={"N":1,"nu":0.05,"lam":4.0}
    eqn=HJB_LQR_Equation_2D(dom,eqn_config)
    solver_params={"initial_lr":0.01,
               "lambda_lr":l_schedule,
               "initial_loss_weigths":[2.0,1.0,3.0,1.0],
               "logging_interval":100,
               "dtype":torch.float32}

    sol=DGM_solver(eqn,solver_params)
    training=sol.train(100000)
    np.save('training.npy',training)
    sol.save_model("./models/LQR_N=1.pth")
