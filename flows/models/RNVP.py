import torch
import torch.nn
from torch import nn
import torch.nn.functional as F
from torch import distributions
import numpy as np

from flows.models.layers import BatchNorm, SNet, TNet


class RealNVP(nn.Module):
    def __init__(self, in_dim, dim_middle, N_layers=3, batch_norm=True, verbose=False, **kwargs):
        super(RealNVP, self).__init__()

        # Create a flow
        # nets:  a function that return a pytocrn neurel network e.g., nn.Sequential, s = nets(), s: dim(X) -> dim(X)
        # nett:  a function that return a pytocrn neurel network e.g., nn.Sequential, t = nett(), t: dim(X) -> dim(X)
        # mask:  a torch.Tensor of size #number_of_coupling_layers x #dim(X)
        # prior: an object from torch.distributions e.g., torch.distributions.MultivariateNormal
        mu = torch.zeros(in_dim)
        sigma = torch.eye(in_dim)


        # creating mask
        onezero = [0, 1] * in_dim
        mask = torch.Tensor([[onezero[:in_dim], onezero[1:in_dim + 1]]] * N_layers)
        mask = mask.view(2 * N_layers, -1)
        print('mask len', len(mask))

        self.prior = distributions.MultivariateNormal(mu, sigma)
        self.in_dim = in_dim
        self.first_layer_features_dim = in_dim-11
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([SNet(dim_in=self.in_dim, dim_middle=dim_middle) for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([TNet(dim_in=self.in_dim, dim_middle=dim_middle) for _ in range(len(mask))])
        self.b = torch.nn.ModuleList([BatchNorm(dim_in=self.in_dim) for _ in range(len(mask))])
        self.batch_norm = batch_norm
        self.verbose = verbose

    def g(self, z):
        # Compute and return g(z) = x,
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer
        # z: a torch.Tensor of shape batchSize x 1 x dim(X)
        # return x: a torch.Tensor of shape batchSize x 1 x dim(X)
        for i, (s, t, b) in enumerate(zip(reversed(self.s), reversed(self.t), reversed(self.b))):
            m = self.mask[-i - 1]
            if self.verbose:
                print('z1', z)
            z = (m * z + (1 - m) * (z - t(m * z)) * (-s(m * z)).exp()) #.detach()
            if self.batch_norm:
                z = (z * (b.sig2 + 1.0e-6).sqrt() + b.mu) #.detach()
            if self.verbose:
                print('z2', z)

        x = z
        return x

    def f(self, x):
        # Compute f(x) = z and log_det_Jakobian of f,
        #    where self.mask[i], self.t[i], self.s[i] define a i-th masked coupling layer
        # x: a torch.Tensor, of shape batchSize x dim(X), is a datapoint
        # return z: a torch.Tensor of shape batchSize x dim(X), a hidden representations
        # return log_det_J: a torch.Tensor of len batchSize

        z = x
        log_det_J = 0
        for s, t, m, b in zip(self.s, self.t, self.mask, self.b):

            if self.batch_norm:
                z, sig2 = b(z)
            s_res = s(m * z)
            z = m * z + (1 - m) * (z * s_res.exp() + t(m * z))

            if self.batch_norm:
                log_det_J += ((1 - m) * s_res - 0.5 * sig2.log()).sum(-1)
            else:
                log_det_J += ((1 - m) * s_res).sum(-1)

        return z, log_det_J

    def log_prob(self, x):
        # Compute and return log p(x)
        # using the change of variable formula and log_det_J computed by f
        # return logp: torch.Tensor of len batchSize
        z, log_det_J = self.f(x)

        logp = -0.5 * np.log(np.pi * 2) - 0.5 * z.pow(2)
        logp = logp.sum(-1)
        # logp = self.prior.log_prob(z.cpu())

        return logp + log_det_J

    def sample(self, K, cuda=True):
        # Draw and return batchSize samples from flow using implementation of g
        # return x: torch.Tensor of shape batchSize x 1 x dim(X)
        shape = torch.Size((K, self.in_dim))
        if cuda:
            e = torch.cuda.FloatTensor(shape)
        else:
            e = torch.FloatTensor(shape)

        torch.randn(shape, out=e)
        x = self.g(e)
        # x = self.f(e)

        return x

    def forward(self, x):
        return


class RealNVP_OneLayer(RealNVP):
    def __init__(self, in_dim, dim_middle, N_layers=3, batch_norm=True, verbose=False, data_b2=[0]*10, **kwargs):
        super(RealNVP_OneLayer, self).__init__(in_dim, dim_middle, N_layers, batch_norm=batch_norm, verbose=verbose)

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


class RealNVP_OneLayer_full(RealNVP):
    def __init__(self, in_dim, dim_middle, N_layers=3, batch_norm=True, verbose=False,
                 data_b2=[0]*10, transPSA=None, **kwargs):
        super(RealNVP_OneLayer_full, self).__init__(in_dim, dim_middle, N_layers,
                                                    batch_norm=batch_norm, verbose=verbose)

        # for prediction
        self.relu = nn.ReLU()
        data_b2 = torch.FloatTensor(data_b2)
        self.b2 = nn.Parameter(data=data_b2, requires_grad=False)
        self.model1 = None
        self.model2 = None
        self.p = 0.5
        self.fix = False
        self.transPSA = transPSA
        if self.transPSA is not None:
            self.PCA_W = nn.Parameter(data=torch.FloatTensor(self.transPSA.components_), requires_grad=False)
            self.PCA_mean = nn.Parameter(data=torch.FloatTensor(self.transPSA.mean_), requires_grad=False)
            self.first_layer_features_dim = self.PCA_W.shape[1]-11

        self.W = None


    def PCA(self, model):
        model = model - self.PCA_mean
        model = model @ torch.t(self.PCA_W)
        return model

    def PCA_inverse(self, model):
        model = model @ self.PCA_W
        model = model + self.PCA_mean
        return model

    def forward(self, input):

        input = input.view(input.size(0), -1)

        if self.transPSA is not None:
            model1 = self.PCA(self.model1)
            model2 = self.PCA(self.model2)
        else:
            model1 = self.model1
            model2 = self.model2

        x = self.f(model1)[0]
        y = self.f(model2)[0]

        if not self.fix:
            if self.training:
                self.p = np.random.uniform(0.4, 0.6)
            else:
                self.p = 0.5

        p = self.p

        a = np.cos(np.pi * p / 2)
        b = np.sin(np.pi * p / 2)
        z = a*x + b*y
        W = self.g(z)
        self.W = W
        if self.transPSA is not None:
            W = self.PCA_inverse(W)

        W1 = W[:, :self.first_layer_features_dim]
        b1 = W[:, self.first_layer_features_dim:self.first_layer_features_dim+1]
        W2 = W[:, self.first_layer_features_dim+1:].transpose(0, 1)

        if self.verbose:
            print('x', x.shape)
            print('W1', W1.shape)
            print('b1', b1.shape)

        x = F.linear(input, W1, b1[:, 0])

        if self.verbose:
            print('x', x.shape)
            print(x.shape)

        x = self.relu(x)

        x = F.linear(x, W2, self.b2)

        if self.verbose:
            print(x.shape)

        return x






