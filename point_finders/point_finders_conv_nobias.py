__all__ = ['PointFinderStepWiseButterflyConv',
            ]

import torch
import numpy as np

from connector import Connector
from tqdm import tqdm
import torch.nn.functional as F

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True
from .point_finders_models import PointFinderSimultaneousData, get_model_from_weights


class PointFinderStepWiseButterflyConv(PointFinderSimultaneousData):
    def __init__(self, model1, model2, architecture, loader, padding=2, kernel_size=5): #2, 5
        self.padding = padding
        self.kernel_size = kernel_size
        super().__init__(model1, model2, architecture, loader)
        self.loader = loader
        self.funcs1 = self.find_feature_maps(self.model1, self.data, self.depth, self.weights_model1)
        self.funcs2 = self.find_feature_maps(self.model2, self.data, self.depth, self.weights_model2)
        self.weights_adjusted = self.adjust_all_weights()


    def get_data(self, loader):
        data = []
        for X, y in loader:
            data.append(X.cpu().data.numpy())
        data = np.concatenate(data)
        return data

    def find_feature_maps(self, model, data, depth, weights_model):
        """find feature maps for 2, 3 ,..., N-2 layers of network"""
        print('finding feature maps')
        model.eval()
        funcs_list = []
        data = torch.tensor(data)  # .cuda()

        for i in range(depth - 1):
            print('i', i, weights_model[i + 1].shape)
            funcs = model(data, N=i)
            if len(funcs.shape) == 4:
                if len(weights_model[i + 1].shape) == 2:
                    print('in')
                    funcs2save = funcs.view(batch, -1)
                else:
                    funcs = F.pad(funcs, (self.padding, self.padding, self.padding, self.padding))
                    batch, chanels, width, high = funcs.shape
                    funcs2save = []
                    for i in range(0, width - self.kernel_size, 2):
                        for j in range(0, high - self.kernel_size, 2):
                            funcs2save.append(funcs[:, :, i:i + self.kernel_size, j:j + self.kernel_size])
                    funcs2save = torch.cat(funcs2save, 0)
                    funcs2save = funcs2save.view(funcs2save.size(0), -1)
            if len(funcs.shape) == 2:
                funcs2save = funcs
            print('funcs2save', funcs2save.shape)
            funcs_list.append(funcs2save.data.cpu().numpy())
        return funcs_list

    def connect_butterflies(self, W10, W20, W11, W11b2,
                            t=0.5, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        Wn1 = getattr(Connector(W11.T, W11b2.T), method)(t=t)[1].T
        return Wn0, Wn1

    def adjust_weights(self, f1, f2, W):
        target_shape = W.shape
        print('target_shape', target_shape)
        if len(target_shape) == 4:
            print('conv')
            print('prod', np.prod(target_shape[1:]))
            W = W.reshape(target_shape[0], np.prod(target_shape[1:]))
        else:
            print('lin')

        print('W, f1', W.shape, f1.shape)
        f_inv2 = np.linalg.pinv(f2.T)
        print('f_inv', f_inv2.shape)
        Wb2 = W @ f1.T @ f_inv2

        if len(target_shape) == 4:
            Wb2 = Wb2.reshape(target_shape)

        return Wb2

    def adjust_all_weights(self, ):
        """find intermidiate weights between \Theta^A and \Theta^B (see the the paper for the notation) """
        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1,
                                                 self.funcs2,
                                                 self.weights_model1[1:-1]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-1])
        return Wb2_list

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        W11b2 = self.weights_adjusted[layer + 1]
        Wn0, Wn1 = self.connect_butterflies(W10, W20, W11, W11b2,
                                            t=t, method=method)
        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        # m.cuda();
        return m
