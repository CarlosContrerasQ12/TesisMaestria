import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as WN
import numpy as np

types=torch.float32
torch.set_default_dtype(types)

class General_FC_net(nn.Module):
    def __init__(self,net_config,din,dout):
        super().__init__()
        default_config={"batch_norm":False,
                        "eps":1e-6,"momentum":0.99,
                        "activation":nn.ReLU(),"final_act":False,"bias":False,
                        "dtype":torch.float32}
        default_config.update(net_config)
        layers=[]

        batch_norm=default_config["batch_norm"]
        lay_sizes=[din]+default_config["int_layers"]+[dout]
        eps=default_config["eps"]
        momentum=default_config["momentum"]
        bias=default_config["bias"]
        dtype=default_config["dtype"]
        activation=default_config["activation"]
        final_act=default_config["final_act"]

        if batch_norm:
            layers.append(nn.BatchNorm1d(lay_sizes[0], eps=eps, momentum=momentum,affine=False,dtype=dtype))
        for i in range(len(lay_sizes)-1):
            layers.append(nn.Linear(lay_sizes[i],lay_sizes[i+1],bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(lay_sizes[i+1], eps=eps, momentum=momentum,affine=False,dtype=dtype))
            if i<len(lay_sizes)-2 or final_act:
                layers.append(activation)
        self.net=nn.Sequential(*layers)

    #@torch.compile
    def forward(self,x):
        return self.net(x)
    
class Global_Model_Deep_BSDE(nn.Module):
    def __init__(self,net_config,eqn):
        super(Global_Model_Deep_BSDE, self).__init__()
        self.net_config = net_config
        self.eqn=eqn
        self.Ndis=net_config["Ndis"]
        self.dt=net_config["dt"]
        self.in_region=net_config["in_region"]

        if self.in_region:
            self.y_0=General_FC_net(net_config["y0_net_config"],self.eqn.dim,1)
            self.z_0=General_FC_net(net_config["z0_net_config"],self.eqn.dim,self.eqn.dim)
        else:
            self.y_0=nn.Parameter(torch.rand(1))
            self.z_0=nn.Parameter((torch.rand((1,self.eqn.dim))*0.2)-0.1)

        self.subnet = [General_FC_net(net_config,self.eqn.dim,self.eqn.dim) for _ in range(self.Ndis-1)]
        self.time_stamp = np.arange(0, self.Ndis) * self.dt
    
    def evaluate_y_0(self,x):
        if self.in_region:
            return self.y_0(x)
        else:
            return self.y_0
        
    def grad_func(self,t,X):
        Xt=torch.tensor(X,dtype=torch.float32)
        Xt=torch.unsqueeze(Xt,axis=0)
        #ind=np.argmin(np.abs(t-self.time_stamp))
        ind=np.argwhere((t-self.time_stamp)<0)[0][0]-1
        self.subnet[ind].eval()
        return self.subnet[ind](Xt)/self.eqn.dim
    
    #@torch.compile
    def forward(self,inputs):
        t,x,dw,states=inputs
        if self.in_region:
            y = self.y_0(x[:,:,0])
            z = self.z_0(x[:,:,0])
        else:
            all_one_vec = torch.ones((dw.shape[0], 1))
            y = all_one_vec * self.y_0
            z = torch.matmul(all_one_vec, self.z_0)

        for i in range(self.Ndis-1):
            y = y - self.dt * (self.eqn.f_torch(self.time_stamp[i], x[:, :, i], y, z,states))+torch.sum(z * dw[:, :, i], 1)
            z = self.subnet[i](x[:, :, i + 1]) / self.eqn.dim
        y = y - self.dt * self.eqn.f_torch(self.time_stamp[-1], x[:, :, -2], y, z,states)+torch.sum(z * dw[:, :, -1], 1)
        return y

class Global_Model_Merged_Deep_BSDE(nn.Module):
    def __init__(self, net_config, eqn):
        super(Global_Model_Merged_Deep_BSDE, self).__init__()
        self.net_config = net_config
        self.eqn=eqn
        self.Ndis=net_config["Ndis"]
        self.dt=net_config["dt"]
        self.in_region=net_config["in_region"]

        self.z_net=General_FC_net(net_config,self.eqn.dim+1,self.eqn.dim)

        if self.in_region:
            self.y_0=General_FC_net(net_config["y0_net_config"],self.eqn.dim,1)
        else:
            self.y_0=nn.Parameter(torch.rand(1))

        self.time_stamp = np.arange(0, self.Ndis) * self.dt
    
    def evaluate_y_0(self,x):
        if self.in_region:
            return self.y_0(x)
        else:
            return self.y_0
        
    def grad_func(self,t,X):
        Xt=torch.tensor(X,dtype=torch.float32)
        inp=torch.hstack((torch.tensor([t]),Xt))
        inp=torch.unsqueeze(inp,axis=0)
        return self.z_net(inp)/self.eqn.dim

    def forward(self, inputs):
        t,x,dw = inputs
        all_one_vec = torch.ones((dw.shape[0], 1))
        if self.in_region:
            y = self.y_0(x[:,:,0])
        else:
            y = all_one_vec * self.y_0

        for t in range(0, self.Ndis):
            inp=torch.hstack((self.time_stamp[t]*all_one_vec,x[:,:,t]))
            z = self.z_net(inp) / self.eqn.dim
            y = y - self.dt * (
                self.eqn.f_tf(self.time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdims=True)
        return y