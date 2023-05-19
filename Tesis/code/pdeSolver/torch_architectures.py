import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as WN



class DGMCell(nn.Module):
    def __init__(self, d, M, growing, weight_norm,sigma=torch.tanh):
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
    def __init__(
            self, d_in, d_out, M=50, L=3,
            growing=False, as_array=True, weight_norm=False,sigma=torch.tanh):
        super().__init__()
        wn = WN if weight_norm else lambda x: x
        self.W0 = wn(nn.Linear(d_in, M))
        self.W1 = wn(nn.Linear(M, d_out))
        self._convert = _set_convert(as_array)
        self.sigma=sigma

        self.layers = []
        for l in range(L):
            self.layers.append(DGMCell(d_in, M, growing, weight_norm,sigma))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, *X):
        X = self._convert(X)
        S = self.sigma(self.W0(X))
        S,_ = self.layers((S, X))
        return self.W1(S).squeeze_(-1)


class DenseNet(nn.Module):
    def __init__(self, d_in, d_out, arch=[30, 30], seed=42):
        super(DenseNet, self).__init__()
        torch.manual_seed(seed)
        self.nn_dims = [d_in] + arch + [d_out]
        self.W = [item for sublist in
                  [[torch.nn.Parameter(torch.randn(sum(self.nn_dims[:i + 1]), self.nn_dims[i + 1],
                                             requires_grad=True) * 0.1),
                    torch.nn.Parameter(torch.zeros(self.nn_dims[i + 1], requires_grad=True))] for
                   i in range(len(self.nn_dims) - 1)]
                  for item in sublist]

        for i, w in enumerate(self.W):
            self.register_parameter('param %d' % i, w)

        #self.optim = pt.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        for i in range(len(self.nn_dims) - 1):
            if i == len(self.nn_dims) - 2:
                x = torch.matmul(x, self.W[2 * i]) + self.W[2 * i + 1]
            else:
                x = torch.cat([x, torch.nn.functional.relu(torch.matmul(x, self.W[2 * i])
                                                     + self.W[2 * i + 1]) ** 2], dim=1)
        return x
