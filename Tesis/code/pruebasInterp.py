import sys
sys.path.append('./pdeSolver')

from domains import *
from equations import *
from torch_solvers import *
import torch
import time

import ipywidgets as widgets
from ipywidgets import interact
from mpl_toolkits import mplot3d
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt


probar=1
if probar==1:
    print("Probando interp")
    dom_config={"total_time":1.0,"pInf":0.4,"pSup":0.6}
    dom=EmptyRoom(dom_config)
    eqn_config={"N":1,"nu":0.05,"lam":4.0}
    eqn=HJB_LQR_Equation_2D(dom,eqn_config)

    solver_params={"initial_lr":0.001,
                "initial_loss_weigths":[1.0,1.0,1.0,1.0],
                "net_size":{"width":30,"depth":4,"weigth_norm":False},
                "logging_interval":10,
                "dtype":torch.float32,
                "dt":0.01,
                "N_max":20,
                "N_samples_per_batch_interior":200,
                "N_samples_per_batch_boundary":40,
                "sample_every":10,
                "alpha":0.9,
                "adaptive_weigths":False}
    sol=Interp_PINN_BSDE_solver(eqn,solver_params)

    sol.train(400)
else:
    print("Probando DGM")
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
               "sample_every":1}

    sol=DGM_solver(eqn,solver_params)
    training=sol.train(5)