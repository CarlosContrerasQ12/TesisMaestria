import numpy as np
import jax
import jax.numpy as jnp
import optax 

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