import time
import numpy as np
import torch
import os

from utils.one_layer_utils import get_model
from utils.train_utils import test_model
from utils.train_utils import train
import torch.nn.functional as F


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def iterate_minibatches(train_data, batchsize, permute=True):
    if permute:
        indices = np.random.permutation(np.arange(len(train_data)))
    else:
        indices = np.arange(len(train_data))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield train_data[ix]


def test_flow(model, architecture, loaders, b, cntr, verbose=True,
              test_sampling=True, test_flow=True, cuda=False, transPSA=None):

        result_s, result_f = None, None

        model.eval()

        if test_sampling:
            if verbose:
                print('sampling testing')
            res = model.sample(K=2000).cpu().data.numpy()
            if transPSA is not None:
                res = transPSA.inverse_transform(res)
            m = get_model(res, b, architecture)
            if cuda:
                m.cuda()
            result_s = test_model(m, loaders, verbose=verbose, cuda=cuda)

        if test_flow:
            if verbose:
                print('flow testing')
            res = cntr.flow_connect(model, cuda=True)[1]
            if transPSA is not None:
                res = transPSA.inverse_transform(res)
            m = get_model(res, b, architecture)
            if cuda:
                m.cuda()
            result_f = test_model(m, loaders, verbose=verbose,  cuda=cuda)

        return result_s, result_f


def train_flow(dataset, model, optimizer,
               batchsize=512, scheduler=None, cuda=True, permute=True):

    model.train()

    if scheduler is not None:
        scheduler.step()

    t = time.time()
    total_loss = 0

    for X in iterate_minibatches(dataset, batchsize, permute=permute):
        if cuda:
            X = torch.FloatTensor(X).cuda()
        else:
            X = torch.FloatTensor(X)
        loss = -model.log_prob(X).mean()  # compute the maximum-likelihood loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss

    total_loss /= (len(dataset) / batchsize)
    print('loss = %.3f' % total_loss, 'time = %.2f' % (time.time() - t))

    return total_loss


def make_models_dataset(dataset):

    models = [dataset[i*2000:(i+1)*2000] for i in range(int(len(dataset)/2000))]
    print('N_models', len(models))
    models_dataset = np.stack(models)

    return models_dataset


def test_bijection(dataset, model, loaders, cuda=True):

    dataset = make_models_dataset(dataset)

    model.eval()
    for X in iterate_minibatches(dataset, batchsize=2, permute=False):

        if cuda:
            model.model1 = torch.FloatTensor(X[0]).cuda()
            model.model2 = torch.FloatTensor(X[1]).cuda()
        else:
            model.model1 = torch.FloatTensor(X[0])
            model.model2 = torch.FloatTensor(X[1])

        test_model(model, loaders, cuda=cuda)


def train_bijection(dataset, model, optimizer, loaders,
                    scheduler=None, cuda=True,
                    verbose=True):

    dataset = make_models_dataset(dataset)
    model.train()

    if scheduler is not None:
        scheduler.step()

    for X in iterate_minibatches(dataset, batchsize=2, permute=True):
        t = time.time()

        if cuda:
            model.model1 = torch.FloatTensor(X[0]).cuda()
            model.model2 = torch.FloatTensor(X[1]).cuda()
        else:
            model.model1 = torch.FloatTensor(X[0])
            model.model2 = torch.FloatTensor(X[1])

        train_res = train(loaders['train'], model, optimizer, criterion=F.cross_entropy, regularizer=None,
                          cuda=cuda)
        if verbose:
            print('p', model.p)
            print(train_res)
            print('time = %.2f' % (time.time() - t))







