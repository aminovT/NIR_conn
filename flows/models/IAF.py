from torch.nn.parameter import Parameter
import math

import numpy as np
import torch
import torch.nn
from torch import nn
import torch.nn.functional as F

from torch import distributions
from flows.models.layers import BatchNorm


class AutoregressiveLinear(nn.Module):
    def __init__(self, dim_in, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = dim_in
        self.out_size = out_size

        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = nn.init.xavier_normal_(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output


class AutoregressiveLinearU(nn.Module):
    def __init__(self, dim_in, out_size, bias=True, ):
        super(AutoregressiveLinearU, self).__init__()

        self.in_size = dim_in
        self.out_size = out_size

        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = nn.init.xavier_normal_(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.triu(1))

        output = input @ self.weight.triu(1)
        if self.bias is not None:
            output += self.bias
        return output


class IAF_block(nn.Module):
    def __init__(self, dim_in, dim_middle, Low=True):
        super(IAF_block, self).__init__()

        self.z_size = dim_in
        self.dim_middle = dim_middle
        self.Low = Low
        self.h = nn.LeakyReLU()

        if Low:
            self.m = nn.Sequential(
                AutoregressiveLinear(self.z_size, self.dim_middle),
                self.h,
                AutoregressiveLinear(self.dim_middle, self.dim_middle),
                self.h,
                AutoregressiveLinear(self.dim_middle, self.z_size)
            )

            self.s = nn.Sequential(
                AutoregressiveLinear(self.z_size, self.dim_middle),
                self.h,
                AutoregressiveLinear(self.dim_middle, self.dim_middle),
                self.h,
                AutoregressiveLinear(self.dim_middle, self.z_size)
            )
        else:
            self.m = nn.Sequential(
                AutoregressiveLinearU(self.z_size, self.dim_middle),
                self.h,
                AutoregressiveLinearU(self.dim_middle, self.dim_middle),
                self.h,
                AutoregressiveLinearU(self.dim_middle, self.z_size)
            )

            self.s = nn.Sequential(
                AutoregressiveLinearU(self.z_size, self.dim_middle),
                self.h,
                AutoregressiveLinearU(self.dim_middle, self.dim_middle),
                self.h,
                AutoregressiveLinearU(self.dim_middle, self.z_size)
            )

    def forward(self, z):

        self.mu_z = self.m(z)
        self.log_sigma_z = self.s(z)
        self.log_sigma_z = torch.clamp(self.log_sigma_z, -5, 5)
        z = self.log_sigma_z.exp() * z + self.mu_z

        return z, self.log_sigma_z


class Flow_IAF(nn.Module):
    def __init__(self, in_dim, dim_middle, N_layers=3, batch_norm=True, verbose=False, **kwargs):
        super(Flow_IAF, self).__init__()

        self.in_dim = in_dim
        self.first_layer_features_dim = in_dim - 11
        self.prior = distributions.MultivariateNormal(torch.zeros(in_dim), torch.eye(in_dim))
        self.len = 2*N_layers
        self.s = torch.nn.ModuleList([IAF_block(dim_in=in_dim, dim_middle=dim_middle, Low=i % 2) for i in range(self.len)])
        self.b = torch.nn.ModuleList([BatchNorm(dim_in=in_dim) for _ in range(self.len)])

        self.verbose = verbose
        self.batch_norm = batch_norm

    def g(self, z):
        z = z.detach()
        for s, b in zip(reversed(self.s), reversed(self.b)):
            if s.Low:
                crange = reversed(range(z.size()[1]))
            else:
                crange = range(z.size()[1])

            for i in crange:

                _, log_sigma = s(z.detach())
                mu = s.mu_z
                if self.verbose:
                    print(i)
                    print('z1', z[:, i])
                z[:, i] = ((z[:, i] - mu[:, i]) * (-log_sigma[:, i]).exp()).detach()
                if self.verbose:
                    print('mu', mu[:, i])
                    print('sigma', (-log_sigma[:, i]).exp())
                    print('z2', z[:, i])

            if self.verbose:
                print('z1-bn', z)
            if self.batch_norm:
                z = z * (b.sig2 + 1.0e-6).sqrt() + b.mu
            if self.verbose:
                print('z2-bn', z)

        x = z
        return x

    def f(self, x):
        z = x
        log_det_J = 0

        for i, (s, b) in enumerate(zip(self.s, self.b)):

            if self.batch_norm:
                z, sig2 = b(z)

            z, log_sigma = s(z)
            if self.batch_norm:
                log_det_J += (log_sigma - 0.5 * sig2.log()).sum(-1)
            else:
                log_det_J += log_sigma.sum(-1)

        return z, log_det_J

    def log_prob(self, x):
        z, log_det_J = self.f(x)

        logp = -0.5 * np.log(np.pi * 2) - 0.5 * z.pow(2)
        logp = logp.sum(-1)

        return logp + log_det_J

    def sample(self, K, cuda=True):
        shape = torch.Size((K, self.in_dim))
        if cuda:
            e = torch.cuda.FloatTensor(shape)
        else:
            e = torch.FloatTensor(shape)
        torch.randn(shape, out=e)
        x = self.g(e)

        return x

    def forward(self, x):
        return



class IAF_OneLayer(Flow_IAF):
    def __init__(self, in_dim, dim_middle, N_layers=3, batch_norm=True, verbose=False, data_b2=[0]*10):
        super(IAF_OneLayer, self).__init__(in_dim, dim_middle, N_layers=N_layers, batch_norm=batch_norm, verbose=verbose)

        # for prediction
        self.relu = nn.ReLU()
        data_b2 = torch.FloatTensor(data_b2)
        self.b2 = nn.Parameter(data=data_b2, requires_grad=False)

    def forward(self, x, K=2000):

        x = x.view(x.size(0), -1)
        W = self.sample(K)
        W1 = W[:, :784]
        b1 = W[:, 784:785]
        W2 = W[:, 785:].transpose(0, 1)

        if self.verbose:
            print('x', x.shape)
            print('W1', W1.shape)
            print('b1', b1.shape)

        x = F.linear(x, W1, b1[:, 0])

        if self.verbose:
            print('x', x.shape)
            print(x.shape)

        x = self.relu(x)
        x = F.linear(x, W2, self.b2)

        if self.verbose:
            print(x.shape)

        return x

class IAF_OneLayer_full(Flow_IAF):
    def __init__(self, in_dim, dim_middle, N_layers=3, batch_norm=True, verbose=False, data_b2=[0]*10,
                 transPSA=None, **kwargs):
        super(IAF_OneLayer_full, self).__init__(in_dim, dim_middle, N_layers, batch_norm=batch_norm, verbose=verbose)

        # for prediction
        self.relu = nn.ReLU()
        data_b2 = torch.FloatTensor(data_b2)
        self.b2 = nn.Parameter(data=data_b2, requires_grad=False)
        self.model1 = None
        self.model2 = None
        self.transPSA = transPSA
        if self.transPSA is not None:
            self.trans = nn.Parameter(data=torch.FloatTensor(self.transPSA.components_), requires_grad=False)
            self.first_layer_features_dim = self.trans.shape[1] - 11

    def forward(self, input):

        input = input.view(input.size(0), -1)
        x = self.f(self.model1)[0]
        y = self.f(self.model2)[0]

        t = 0.5
        a = np.cos(np.pi * t / 2)
        b = np.sin(np.pi * t / 2)
        z = a*x + b*y
        W = self.g(z)

        if self.transPSA is not None:
            W = W @ self.trans

        W1 = W[:, :self.first_layer_features_dim]
        b1 = W[:, self.first_layer_features_dim:self.first_layer_features_dim+1]
        W2 = W[:, self.first_layer_features_dim+1:].transpose(0, 1)

        if self.verbose:
            print('x', x.shape)
            print('W1', W1.shape)
            print('b1', b1.shape)

        print('W1', W1.shape)
        print('input', input.shape)
        x = F.linear(input, W1, b1[:, 0])

        if self.verbose:
            print('x', x.shape)
            print(x.shape)

        x = self.relu(x)

        x = F.linear(x, W2, self.b2)

        if self.verbose:
            print(x.shape)

        return x