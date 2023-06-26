from domains import *
from equations import *
from torch_solvers import *
import torch

N=50
ps=15
dom_config={"Domain":'FreeSpace'}
dom=FreeSpace(dom_config)
eqn_config={"N":N,"terminal_time":1.0,"nu":1.0,"lam":1.0}
eqn=HJB_LQR_Equation(dom,eqn_config)
print("Probando Normal Deep BSDE")
solver_params={"initial_lr":0.01,
               "net_config":{"net_type":'Normal',
                            "int_layers":[eqn.dim+10,eqn.dim+10],
                          "batch_norm":False,
                          "y0_net_config":{"int_layers":[N+ps,N+ps],"bias":True},
                          "z0_net_config":{"int_layers":[N+ps,N+ps],"bias":True}},
               "logging_interval":100,
               "dtype":torch.float32,
               "Ntdis":60,
               "Nsamp":100,
               "batch_size":100,
               "sample_every":1,
               "in_region":False}
sol_normal=Deep_BSDE_Solver(eqn, solver_params)
training_normal=sol_normal.train(1000)