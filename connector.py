from pylab import rcParams

import multiprocessing
from sklearn.decomposition import PCA, KernelPCA
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
from scipy.stats import norm
import seaborn as sns


class distribution():
    def __init__(self, X):
        self.sorted_X = np.sort(X)
        self.uniform = np.arange(len(X) + 1) / len(X)

    def F(self, x):

        b = self.sorted_X >= x
        ind = np.argmax(b)
        if ~b[-1] and ind == 0:
            return 1
        elif b[0] and ind == 0:
            return 0
        else:
            delta = (x - self.sorted_X[ind - 1]) / (self.sorted_X[ind] - self.sorted_X[ind - 1])
            return self.uniform[ind] + delta * (1 / len(self.sorted_X))

    def inv_F(self, u):

        b = self.uniform[1:] >= u
        ind = np.argmax(b)
        if b[-1] and ind == (len(self.sorted_X) - 1):
            return self.sorted_X[-1]
        elif b[0] and ind == 0:
            return self.sorted_X[0]
        else:
            delta = (u - self.uniform[ind]) / (1 / len(self.sorted_X))
            return self.sorted_X[ind - 1] + delta * (self.sorted_X[ind] - self.sorted_X[ind - 1])



def f(Z):
    X, All = Z
    distr = distribution(All)
    u = []
    for x in X:
        u.append(distr.F(x))

    return np.array(u)


def g(Z):
    X, U = Z
    distr = distribution(X)
    u = []
    for x in U:
        u.append(distr.inv_F(x))

    return np.array(u)

def get_par(S, mean):
    new_S = S - mean
    cov = new_S.T @ new_S
    U, sigma, V = np.linalg.svd(cov)
    return U, sigma

def get_new_S(S, U, mean):
    new = (S-mean)@U
    return new

def get_inverse_new_S(V, U, mean):
    S = V@np.linalg.inv(U)+mean
    return S


def constract_U(S1, S2, S, ebs=1e-16):
    mean = S.mean(0)

    SPS = S
    SPS1 = S1
    SPS2 = S2

    U1, sigma1 = get_par(SPS1, mean)
    U, _ = get_par(SPS, mean)

    U_temp = U1 * (1 / np.sqrt(sigma1 + ebs))

    S2_temp = (SPS2 - mean) @ U_temp
    U2_temp, sigma2_temp = get_par(S2_temp, 0)
    U = U_temp @ U2_temp

    return U


class Connector():
    def __init__(self, S1, S2, random_state=1):
        self.S1 = np.array(S1)
        self.S2 = np.array(S2)
        self.S = np.vstack([S1, S2])
        self.third_cumulants = None
        self.random_state = random_state
        self.transPSA = None
        self.c = 0.


    def transform(self, use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                  use_third_cumulant=False, t=0.5, K=2, use_mean=False,  flow=None, use_flow=False,
                  method='Arc', l2normalize=False,
                  simul_diag=False, two_simul_diag=False, clip_value=False, use_third_cumulant2=False,
                  cuda=False):

        """""
            method: str (Lin, Arc, Slerp) 
        """""

        if isinstance(t, (int, float)):
            points = [t]
        else:
            points = t

        intermediate_samples = []

        S = copy.deepcopy(self.S)
        S1 = copy.deepcopy(self.S1)
        S2 = copy.deepcopy(self.S2)

        if simul_diag:
            U = constract_U(S1, S2, S)
            mean = self.S.mean(0)
            S1 = get_new_S(S1, U, mean)
            S2 = get_new_S(S2, U, mean)
            S = get_new_S(S, U, mean)

        if two_simul_diag:
            U1 = constract_U(S1, S2, S)
            U2 = constract_U(S2, S1, S)
            U = 0.5*(U1+U2)
            mean = self.S.mean(0)
            S1 = get_new_S(S1, U, mean)
            S2 = get_new_S(S2, U, mean)
            S = get_new_S(S, U, mean)

        if use_flow:
            if flow is not None:
                flow.eval()
                if cuda:
                    S = flow.f(torch.FloatTensor(S).cuda())[0].data.cpu().numpy()
                    S1 = flow.f(torch.FloatTensor(S1).cuda())[0].data.cpu().numpy()
                    S2 = flow.f(torch.FloatTensor(S2).cuda())[0].data.cpu().numpy()
                else:
                    S = flow.f(torch.FloatTensor(S))[0].data.cpu().numpy()
                    S1 = flow.f(torch.FloatTensor(S1))[0].data.cpu().numpy()
                    S2 = flow.f(torch.FloatTensor(S2))[0].data.cpu().numpy()

        if use_mean:
            mean = S.mean(0)
            S = S - mean
            S1 = S1 - mean
            S2 = S2 - mean

        if use_PSA:
            # if self.transPSA is None:
            self.transPSA = PCA(n_components=K, svd_solver='full', random_state=self.random_state)
            # self.transPSA = KernelPCA(n_components=K, random_state=self.random_state, fit_inverse_transform=True)
            S = self.transPSA.fit_transform(S)
            S1 = self.transPSA.transform(S1)
            S2 = self.transPSA.transform(S2)

        if use_std:
            std = S.std(0)
            S = S / std
            S1 = S1 / std
            S2 = S2 / std

        if l2normalize:
            N1 = np.linalg.norm(S1)
            N2 = np.linalg.norm(S2)
            NORM = (N1+N2)/2
            S1 = S1 / N1
            S2 = S2 / N2

        if inverse:
            if use_one:
                TR1 = S
                TR2 = S
                TRF = S
            else:
                TR1 = S1
                TR2 = S2
                TRF = S

            pool = multiprocessing.Pool(1)
            res1 = np.array(pool.map(f, zip(S1.T, TR1.T))).T
            res2 = np.array(pool.map(f, zip(S2.T, TR2.T))).T

            rv = norm()

            nres1 = np.clip(rv.ppf(res1), -10, 10)
            nres2 = np.clip(rv.ppf(res2), -10, 10)
        else:
            nres1, nres2 = S1, S2

        if use_third_cumulant or use_third_cumulant2:
            third_cumulants = self.compute_3cumulant(S)
            self.third_cumulants = third_cumulants

        if use_third_cumulant2:
            S_normal = np.random.multivariate_normal(mean=np.zeros(S1.shape[1]), cov=np.eye(S1.shape[1]), size=len(S1))
            delta = self.compute_flow_delta(third_cumulants, S_normal)
            nres1 -= delta
            nres2 -= delta

        for t in points:
            if method == 'Lin':
                a = 1-t
                b = t
            elif method == 'Slerp':
                print(np.dot(S1, S2).shape)
                print(np.arccos(np.dot(S1, S2).shape))
                print(np.linalg.norm(S1).shape)
                phi = np.arccos(np.dot(S1, S2)/(np.linalg.norm(S1)*np.linalg.norm(S2)))
                sin = np.sin(phi)
                a = np.sin((1-t)*phi)/sin
                b = np.sin(t*phi)/sin
            else:
                a = np.cos(np.pi * t / 2)
                b = np.sin(np.pi * t / 2)

            nfin = a * nres1 + b * nres2

            if use_third_cumulant:
                delta = self.compute_3cumulant_delta(third_cumulants, nres1, nres2, a, b)
                nfin = nfin + delta


            if use_third_cumulant2:
                nfin = nfin + delta

            if inverse:
                ufin = rv.cdf(nfin)
                fin = np.array(pool.map(g, zip(TRF.T, ufin.T))).T
            else:
                fin = nfin

            if use_std:
                fin = std * fin

            if l2normalize:
                fin = NORM * fin

            if use_PSA:
                fin = self.transPSA.inverse_transform(fin)

            if use_mean:
                fin += mean

            if use_flow:
                if cuda:
                    fin = flow.g(torch.FloatTensor(fin).cuda()).data.cpu().numpy()
                else:
                    fin = flow.g(torch.FloatTensor(fin)).data.cpu().numpy()

            if simul_diag:
                fin = get_inverse_new_S(fin, U, mean)

            if two_simul_diag:
                fin = get_inverse_new_S(fin, U, mean)

            if show and inverse:
                ind = 0
                rcParams['figure.dpi'] = 50
                plt.hist(self.S2[:, ind], bins=30);
                plt.show()
                plt.hist(res2[:, ind], bins=30);
                plt.show()
                plt.hist(nres2[:, ind], bins=50);
                plt.show()
                plt.hist(nfin[:, ind], bins=50);
                plt.show()
                plt.hist(ufin[:, ind], bins=50);
                plt.show()
                plt.hist(fin[:, ind], bins=50);
                plt.show()

            if clip_value:
                for ind, (r1, c1) in enumerate(zip(fin.T, self.S.T)):
                    fin.T[ind] = np.clip(r1, c1.min(), c1.max())

            intermediate_samples.append(fin)

        intermediate_samples = np.array(intermediate_samples)

        return np.concatenate([[self.S1], intermediate_samples, [self.S2]])

    def slerp_connect(self, t=0.5, ):
        return self.transform(use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_mean=True, use_flow=False,
                              method='Slerp', l2normalize=True)

    def arc_connect(self, t=0.5, ):
        return self.transform(use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_mean=True, use_flow=False)

    def arc_connect_nomean(self, t=0.5, ):
        return self.transform(use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_mean=False, use_flow=False)

    def arc_connect_PCA(self, t=0.5, K=None):
        if K is None:
            K = min(self.S.shape[1], self.S.shape[0])
        return self.transform(use_PSA=True, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_mean=True, use_flow=False, K=K)

    def lin_mean_connect(self, t=0.5, ):
        return self.transform(use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_mean=True, use_flow=False, method='Lin')

    def lin_connect(self, t=0.5, ):
        return self.transform(use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_mean=False, use_flow=False, method='Lin')


    def inverse_connect(self, t=0.5, K=None):
        if K is None:
            K = min(self.S.shape[1], self.S.shape[0])
        return self.transform(use_PSA=False, use_one=True, show=False, use_std=False, inverse=True,
                              use_third_cumulant=False, t=t, K=K, use_mean=False,  use_flow=False)

    def inverse_connect_PCA(self, t=0.5, K=None):
        if K is None:
            K = min(self.S.shape[1], self.S.shape[0])
        return self.transform(use_PSA=True, use_one=True, show=False, use_std=False, inverse=True,
                              use_third_cumulant=False, t=t, K=K, use_mean=False,  use_flow=False)

    def simul_diag_connect(self, t=0.5, K=None):

        if K is None:
            K = min(self.S.shape[1], self.S.shape[0])
        return self.transform(use_PSA=False, use_one=True, show=False, use_std=False, inverse=True,
                              use_third_cumulant=False, t=t, K=K, use_mean=False,
                              use_flow=False, simul_diag=True)

    def third_cumulant_connect(self, t=0.5, K=None):
        if K is None:
            K = min(self.S.shape[1], self.S.shape[0])
        return self.transform(use_PSA=True, use_one=False, show=False, use_std=True, inverse=False,
                              use_third_cumulant=True, t=t, K=K, use_mean=True, use_flow=False, clip_value=True)

    def flow_connect(self, model, t=0.5, cuda=False):
        return self.transform(use_PSA=False, use_one=False, show=False, use_std=False, inverse=False,
                              use_third_cumulant=False, t=t, use_flow=True, use_mean=False, flow=model,
                              cuda=cuda, clip_value=False)

    def compute_3cumulant_delta(self, third_cumulants, nres1, nres2, a, b):
        first, c = self.compute_coef_flow(third_cumulants)
        print('c', np.abs(c).sum(), (np.abs(c)>0).sum())
        delta = []
        S_normal = np.random.multivariate_normal(mean=np.zeros(nres1.shape[1]), cov=np.eye(nres1.shape[1]), size=len(nres1))

        for n1 in range(nres1.shape[1]):
            second = 0
            for n2 in range(nres1.shape[1]):
                for n3 in range(nres1.shape[1]):
                    vsp = (a * nres1[:, n2] + b * nres2[:, n2]) * (a * nres1[:, n3] + b * nres2[:, n3]) \
                          -a*nres1[:, n2]*nres1[:, n3]-b*nres2[:, n2]*nres2[:, n3]
                          # -(a+b)*(S_normal[:, n2]*S_normal[:, n3])

                    second += c[n1, n2, n3] * vsp

            delta.append(-first[n1]*(1-a-b) + second)
        delta = np.array(delta).T
        delta_fin = delta #-delta_normal
        return delta_fin

    def compute_flow_delta(self, third_cumulants, nres1):
        first, c = self.compute_coef_flow(third_cumulants)
        delta = []
        for n1 in range(nres1.shape[1]):
            second = 0
            for n2 in range(nres1.shape[1]):
                for n3 in range(nres1.shape[1]):
                    vsp = nres1[:, n2] * nres1[:, n3]
                    second += c[n1, n2, n3] * vsp

            delta.append(-first[n1] + second)
        delta = np.array(delta).T
        return delta

    def compute_coef_flow(self, third_cumulants):
        c = np.zeros(third_cumulants.shape)
        for n1 in range(third_cumulants.shape[1]):
            for n2 in range(third_cumulants.shape[1]):
                for n3 in range(third_cumulants.shape[1]):
                    # if n1 != n2 == n3:
                    #     continue
                    #     # c[n1, n2, n3] = 1/2*third_cumulants[n1, n2, n3]
                    # elif n1 == n2 != n3:
                    #     c[n1, n2, n3] = 1 / 2 * third_cumulants[n1, n2, n3]
                    #     continue
                    # elif n1 == n3 != n2:
                    #     continue
                    # else:
                    c[n1, n2, n3] = 1/6*third_cumulants[n1, n2, n3]

        c = np.where(np.abs(c) < self.c, 0, c)
        f = np.zeros(third_cumulants.shape[1])
        for n1 in range(third_cumulants.shape[1]):
            for n2 in range(third_cumulants.shape[1]):
                f[n1] += c[n1, n2, n2]

        return f, c

    def compute_3cumulant(self, S_centered, use_mean=False):
        if use_mean:
            S_centered = S_centered - S_centered.mean(0)

        if S_centered.shape[1]>3000:
            S_centered = S_centered[:, ::4]

        pre_cov = np.expand_dims(S_centered, axis=-1) @ np.expand_dims(S_centered, axis=1)
        third_cumulants = []

        for i in range(S_centered.shape[1]):
            third_cumulants1 = pre_cov.T[i] @ S_centered
            third_cumulants.append(third_cumulants1)

        return np.array(third_cumulants) / len(S_centered)

    def stat(self, noisy_normals, mean=True, cov=True, third_cumulant=True, verbose=False):

        to_return = {'mean': [], 'cov': [], 'third_cumulant': []}
        if mean:
            for noisy_normal in noisy_normals:
                m = noisy_normal.mean(0)
                if verbose:
                    print('mean', m)
                to_return['mean'].append(m)
        if cov:
            for noisy_normal in noisy_normals:
                noisy_normal = noisy_normal - noisy_normal.mean(0)
                c = noisy_normal.T @ noisy_normal / len(noisy_normal)
                if verbose:
                    print('cov', c)
                to_return['cov'].append(c)
        if third_cumulant:
            for noisy_normal in noisy_normals:
                noisy_normal = noisy_normal - noisy_normal.mean(0)
                third_c = self.compute_3cumulant(noisy_normal, use_mean=False)
                if verbose:
                    print('third cumulant', third_c)
                to_return['third_cumulant'].append(third_c)

        return to_return

    def show_samples(self, points, show="hex", limits=False, ind1=0, ind2=1, title=True, set_limits=None):
        rcParams['figure.figsize'] = 16, 12
        fig, axes = plt.subplots(1, len(points), )
        if len(points) == 1:
            axes = [axes]
        time = np.linspace(0, 1, len(points))
        if show == "points":
            if limits:
                for a in axes:
                    a.set_xlim([points[0][:, ind1].min(), points[0][:, ind1].max()])
                    a.set_ylim([points[0][:, 1].min(), points[0][:, 1].max()])

            if set_limits is not None:
                for a in axes:
                    a.set_xlim(set_limits['min'][0], set_limits['max'][0])
                    a.set_ylim(set_limits['min'][1], set_limits['max'][1])

            for i in range(len(points)):
                if title:
                    axes[i].set_title('Samples {0:2d}'.format(i) + ' t {0:.2f}'.format(time[i]))
                axes[i].scatter(points[i][:, ind1], points[i][:, ind2])
            plt.show()
        elif show == "hex":
            if limits:
                for a in axes:
                    a.set_xlim([points[0][:, ind1].min(), points[0][:, ind1].max()])
                    a.set_ylim([points[0][:, 1].min(), points[0][:, 1].max()])

            if set_limits is not None:
                for a in axes:
                    a.set_xlim(set_limits['min'][0], set_limits['max'][0])
                    a.set_ylim(set_limits['min'][1], set_limits['max'][1])

            for i in range(len(points)):
                if title:
                    axes[i].set_title('Samples {0:2d}'.format(i)+' t {0:.2f}'.format(time[i]))
                axes[i].hexbin(points[i][:, ind1], points[i][:, ind2], gridsize=10, cmap=plt.cm.BuGn_r)
            plt.show()
        elif show == "kde":
            rcParams['figure.figsize'] = 8, 6
            fig, axes = plt.subplots(1, 1, )
            if limits:
                axes.set_xlim([points[0][:, ind1].min(), points[0][:, ind1].max()])
                axes.set_ylim([points[0][:, ind2].min(), points[0][:, ind2].max()])
            for i in range(len(points)):
                sns.kdeplot(points[i][:, ind1], points[:, ind2], shade=True, cbar=True)
                plt.show()

        return fig
