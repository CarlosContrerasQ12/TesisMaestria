import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as WN
import numpy as np

types=torch.float32
torch.set_default_dtype(types)

class DGMCell(nn.Module):
    def __init__(self, d, M, growing, weight_norm,sigma):
        super().__init__()
        wn = WN if weight_norm else lambda x: x

        self.Uz = wn(nn.Linear(d, M, bias=False))
        self.Ug = wn(nn.Linear(d, M, bias=False))
        self.Ur = wn(nn.Linear(d, M, bias=False))
        self.Uh = wn(nn.Linear(d, M, bias=False))

        self.Wz = wn(nn.Linear(M, M))
        self.Wg = wn(nn.Linear(M, M))
        self.Wr = wn(nn.Linear(M, M))
        self.Wh = wn(nn.Linear(M, M))

        self.A = (lambda x: x) if growing else sigma
        self.sigma=sigma

    def forward(self, SX):
        S, X = SX
        Z = self.sigma(self.Uz(X) + self.Wz(S))
        G = self.sigma(self.Ug(X) + self.Wg(S))
        R = self.sigma(self.Ur(X) + self.Wr(S))
        H = self.A(self.Uh(X) + self.Wh(S*R))
        S = (1-G)*H + Z*S

        return S, X


def _set_convert(flag):
    if flag: return lambda X: X[0]
    return lambda X: torch.stack(X, -1)


class ResNetLikeDGM(nn.Module):
    """
    DGM algorithm from https://arxiv.org/pdf/1708.07469.pdf
    Args:
    -----
    d_in and d_out- input and ouput dimensions of the problem
    M - layers' width
    L - recurrency depth
    """
    def __init__(self,net_config,eqn):
        super().__init__()
        default_config={"int_layers":[50,50,50],
                        "growing":False,
                        "as_array":True,"weight_norm":False,
                        "activation":torch.tanh,
                        "dtype":torch.float32}
        default_config.update(net_config)

        d_in=eqn.dim+1
        d_out=1

        growing=default_config["growing"]
        weight_norm=default_config["weight_norm"]
        as_array=default_config["as_array"]
        M=default_config["int_layers"][0]
        sigma=default_config["activation"]
        int_layers=default_config["int_layers"]

        wn = WN if weight_norm else lambda x: x
        self.W0 = wn(nn.Linear(d_in, M))
        self.W1 = wn(nn.Linear(M, d_out))
        self._convert = _set_convert(as_array)
        self.sigma=sigma

        self.layers = []
        for l in int_layers:
            self.layers.append(DGMCell(d_in, l, growing, weight_norm,sigma))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, *X):
        X = self._convert(X)
        S = self.sigma(self.W0(X))
        S,_ = self.layers((S, X))
        return self.W1(S).squeeze_(-1)


class DenseNet(nn.Module):
    def __init__(self, net_config,eqn):
        super(DenseNet, self).__init__()

        default_config={"int_layers":[30,30],
                        "activation":torch.relu,
                        "dtype":torch.float32,
                        "seed":42}
        default_config.update(net_config)

        d_in=eqn.dim+1
        d_out=1

        sigma=default_config["activation"]
        int_layers=default_config["int_layers"]
        seed=default_config["seed"]

        torch.manual_seed(seed)
        self.nn_dims = [d_in] + int_layers + [d_out]
        self.W = [item for sublist in
                  [[torch.nn.Parameter(torch.randn(sum(self.nn_dims[:i + 1]), self.nn_dims[i + 1],
                                             requires_grad=True) * 0.1),
                    torch.nn.Parameter(torch.zeros(self.nn_dims[i + 1], requires_grad=True))] for
                   i in range(len(self.nn_dims) - 1)]
                  for item in sublist]

        for i, w in enumerate(self.W):
            self.register_parameter('param %d' % i, w)
        self.sigma=sigma

    def forward(self, x):
        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                x = torch.matmul(x, self.W[2 * i]) + self.W[2 * i + 1]
            else:
                x = torch.cat([x, self.sigma(torch.matmul(x, self.W[2 * i])
                                                     + self.W[2 * i + 1]) ** 2], dim=1)
        return x

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

    def forward(self,x):
        return self.net(x)


class Global_Model_Deep_BSDE(nn.Module):
    def __init__(self, net_config,eqn):
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

        self.subnet = [General_FC_net(net_config,self.eqn.dim,self.eqn.dim) for _ in range(self.Ndis-2)]
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


    def forward(self, inputs):
        dw, x = inputs
        if self.in_region:
            y = self.y_0(x[:,:,0])
            z = self.z_0(x[:,:,0])
        else:
            all_one_vec = torch.ones((dw.shape[0], 1))
            y = all_one_vec * self.y_0
            z = torch.matmul(all_one_vec, self.z_0)

        for t in range(0, self.Ndis-2):
            y = y - self.dt * (
                self.eqn.f_tf(self.time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdims=True)
            z = self.subnet[t](x[:, :, t + 1]) / self.eqn.dim
        # terminal time
        y = y - self.dt * self.eqn.f_tf(self.time_stamp[-1], x[:, :, -2], y, z) + \
            torch.sum(z * dw[:, :, -1], 1, keepdims=True)
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
        dw, x = inputs
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
    

class Global_Model_Merged_Residual_Deep_BSDE(nn.Module):
    def __init__(self, net_config, eqn):
        super(Global_Model_Merged_Residual_Deep_BSDE, self).__init__()
        self.net_config = net_config
        self.eqn=eqn
        self.Ndis=net_config["Ndis"]
        self.dt=net_config["dt"]
        self.in_region=net_config["in_region"]

        self.z_net=General_FC_Residual_net(net_config,self.eqn.dim+1+1+1,self.eqn.dim)

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

    def forward(self, inputs):
        dw, x = inputs
        all_one_vec = torch.ones((dw.shape[0], 1),dtype=torch.float32)
        if self.in_region:
            y = self.y_0(x[:,:,0])
        else:
            y = all_one_vec * self.y_0

        for t in range(0, self.Ndis):
            
            inp=torch.hstack((self.time_stamp[t]*all_one_vec,y,self.eqn.g_tf(x[:,:,t]),x[:,:,t]))
            z = self.z_net(inp) / self.eqn.dim
            kss=self.eqn.f_tf(self.time_stamp[t], x[:, :, t], y, z)
            y = y - self.dt * (
                self.eqn.f_tf(self.time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keepdims=True)
        return y

class General_FC_Residual_net(nn.Module):
    def __init__(self, net_config,din,dout):
        super().__init__()
        default_config={"batch_norm":False,
                        "eps":1e-6,"momentum":0.99,
                        "activation":nn.ELU(),"final_act":False,"bias":False,
                        "dtype":torch.float32}
        default_config.update(net_config)

        lay_sizes=[din]+default_config["int_layers"]+[dout]
        self.num_int_layers=len(lay_sizes)-2
        self.bias=default_config["bias"]
        dtype=default_config["dtype"]
        self.activation=default_config["activation"]
        self.final_act=default_config["final_act"]
        self.layers=torch.nn.ModuleList()

        for i in range(len(lay_sizes)-1):
            self.layers.append(nn.Linear(lay_sizes[i],lay_sizes[i+1],bias=self.bias,dtype=dtype))


    def forward(self,x):
        out1=self.activation(self.layers[0](x))
        out2=out1.clone()
        for i in range(1,self.num_int_layers):
            out2=self.activation(self.layers[i](out2))
        if self.final_act:
            return self.activation(self.layers[-1](out1+out2))
        else:
            return self.layers[-1](out1+out2)
        
class Global_Raissi_Net(nn.Module):
    def __init__(self,net_config,eqn):
        super(Global_Raissi_Net, self).__init__()
        self.eqn=eqn
        self.model=General_FC_net(net_config,self.eqn.dim+1,self.eqn.dim)

    def forward(self,tx):
        return self.model(tx)