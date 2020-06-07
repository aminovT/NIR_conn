from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import os
import sys
import numpy as np

from sklearn import datasets
import pickle

import flows
from flows.models.utils import save_checkpoint, train_flow, train_bijection, test_bijection, test_flow
from connector import Connector


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='flow_models/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='make_moons', metavar='DATASET',
                    help='dataset name (default: make_moons)')
parser.add_argument('--dataset_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--model_dataset_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--save_freq', type=int, default=1, metavar='N',
                    help='save frequency')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay')
parser.add_argument('--scheduler_step_size', type=int, default=20)
parser.add_argument('--test_every', type=int, default=1)


parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--cuda', action='store_true')


parser.add_argument('--bijection', action='store_true')
parser.add_argument('--sample_t', action='store_true')

parser.add_argument('--flow', action='store_true')
parser.add_argument('--test_flow', action='store_true')
parser.add_argument('--base_model', type=str, default="LinearOneLayer")
parser.add_argument('--permute', action='store_true')

parser.add_argument('--dim_middle', type=int, default=100)
parser.add_argument('--N_layers', type=int, default=3)
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--PCA', action='store_true')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

if args.cuda:
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)

# dataset
b = None
transformation = None
n_components = None
if args.dataset_path is not None:
    with open(args.dataset_path, 'rb') as handle:
        load = pickle.load(handle)
        dataset, B = load

        print('dataset', dataset.shape)
        train_dataset, test_dataset = dataset[:-8000], dataset[-8000:]
        b = np.array(B).mean(0)
        print('b', b.shape)
else:
    make_dataset = getattr(datasets, args.dataset)
    dataset = make_dataset(n_samples=45000, noise=.05, random_state=args.seed)[0]
    train_dataset, test_dataset = dataset[:-5000], dataset[-5000:]

architecture = getattr(flows.models, args.model)
in_dim = dataset.shape[1]
if args.PCA:
    in_dim = n_components

model = architecture(in_dim=in_dim, dim_middle=args.dim_middle, N_layers=args.N_layers,
                     batch_norm=args.batch_norm, data_b2=b, transPSA=transformation)

if args.cuda:
    model.cuda()

trainable_parametrs = filter(lambda param: param.requires_grad,
                             model.parameters())
trainable_parametrs = model.parameters()
optimizer = torch.optim.Adam(trainable_parametrs, lr=args.lr,
                             weight_decay=args.wd)

scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.9)

if args.test_flow:

    import models
    base_architecture = getattr(models, args.base_model)

    W_models = [test_dataset[:2000], test_dataset[2000:4000]]
    cntr = Connector(*W_models)

if args.bijection or args.test_flow:
    import data
    if args.test_flow:
        batch_size = 1024
    else:
        batch_size = args.batch_size
    loaders, num_classes = data.loaders(
        args.dataset,
        args.model_dataset_path,
        batch_size,
        1,
        "VGG",
        True)


for epoch in range(args.epochs):

    if args.flow:
        print('trainig flow...', epoch)
        if args.test_flow:
            if epoch % args.test_every == 0:
                print('flow testing 1 ')
                test_flow(model, base_architecture, loaders, torch.FloatTensor(b),
                          cntr, verbose=True, test_sampling=True, test_flow=False,
                          cuda=args.cuda, transPSA=transformation)

        train_flow(train_dataset, model, optimizer,
                   batchsize=args.batch_size, cuda=args.cuda,  scheduler=scheduler, permute=args.permute)

    if args.bijection:
        print('trainig bijection...', epoch)
        model.fix = not args.sample_t
        train_bijection(train_dataset, model, optimizer, loaders,
                        scheduler=scheduler, cuda=args.cuda,
                        verbose=True)
        if epoch % args.test_every == 0:
            test_bijection(test_dataset, model, loaders, cuda=args.cuda)

    if epoch % args.save_freq == 0:
        save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

save_checkpoint(
    args.dir,
    args.epochs,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)