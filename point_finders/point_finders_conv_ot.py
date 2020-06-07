__all__ = ['PointFinderStepWiseButterflyConvWBiasOT',
           'PointFinderStepWiseButterflyConvWBiasOT2',
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
from .point_finders_models import get_model_from_weights
from .point_finders_conv import PointFinderStepWiseButterflyConvWBias
import ot
import ot.plot

class PointFinderStepWiseButterflyConvWBiasOT(PointFinderStepWiseButterflyConvWBias):
    def __init__(self, model1, model2, architecture, loader,): #2, 5
        super().__init__(model1, model2, architecture, loader)
        self.GO = []
        self.M = []

    def solve_optimal_transport_problem(self, weights1, weights2):
        n = len(weights1)
        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
        # loss matrix
        M = ot.dist(weights1.reshape(weights1.shape[0],-1), weights2.reshape(weights2.shape[0],-1))
        M /= M.max()
        self.M.append(M)
        GO = ot.emd(a, b, M)
        self.GO.append(GO)
        return GO


    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[0::2][layer + 1]
        W20 = self.weights_model2[0::2][layer]
        W10 = self.weights_adjusted[layer]
        W11b2 = self.weights_adjusted[layer + 1]
        B10 = self.weights_model1[1::2][layer]
        B20 = self.weights_model2[1::2][layer]


        GO = self.solve_optimal_transport_problem(W10, W20)
        indices = np.argmax(GO, axis=-1)
        W20 = W20[indices]
        B20 = B20[indices]

        Wn0, Bn0, Wn1 = self.connect_butterflies(W10, W20, B10, B20, W11, W11b2,
                                            t=t, method=method)
        weights_model_t = self.weights_model2[:2*layer] + [Wn0, Bn0, Wn1] + self.weights_model1[2*layer + 3:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        return m


class PointFinderStepWiseButterflyConvWBiasOT2(PointFinderStepWiseButterflyConvWBiasOT):
    def __init__(self, model1, model2, architecture, loader,): #2, 5
        super().__init__(model1, model2, architecture, loader)

    def butterfly_weights(self, W1, B1, W2):
        samples = np.hstack([W1.reshape(W1.shape[0], -1), B1[:, None], self.transpose(W2).reshape(W1.shape[0], -1)])
        # print('samples', samples.shape)
        return samples

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[0::2][layer + 1]
        W20 = self.weights_model2[0::2][layer]
        W10 = self.weights_adjusted[layer]
        W11b2 = self.weights_adjusted[layer + 1]
        B10 = self.weights_model1[1::2][layer]
        B20 = self.weights_model2[1::2][layer]

        W1_butterflyed = self.butterfly_weights(W10, B10, W11)
        W2_butterflyed = self.butterfly_weights(W20, B20, W11b2)
        GO = self.solve_optimal_transport_problem(W1_butterflyed, W2_butterflyed)
        indices = np.argmax(GO, axis=-1)
        W20 = W20[indices]
        B20 = B20[indices]

        Wn0, Bn0, Wn1 = self.connect_butterflies(W10, W20, B10, B20, W11, W11b2,
                                            t=t, method=method)
        weights_model_t = self.weights_model2[:2*layer] + [Wn0, Bn0, Wn1] + self.weights_model1[2*layer + 3:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        return m