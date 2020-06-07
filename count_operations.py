import argparse
import numpy as np
import os
import torch

import data as dateset
import models
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='DNN curve evaluation')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--num_models', type=int, default=7, metavar='NM',
                    help='number of models (default: 7)')
parser.add_argument('--num_layers', type=int, default=16, metavar='NL',
                    help='number of num_layers in a model')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)

print('computing number of operations in ensemble of {} {} models via WA method'.format(args.num_models, args.model,))

loaders, num_classes = dateset.loaders(
    args.dataset,
    args.data_path,
    1,
    0,
    args.transform,
    train_random=False,
    shuffle_train=False,
)

num_classes = int(num_classes)
architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])

for X, y in loaders['test']:
    break


pixel_resolution = []
print('getting features for each layer')

for i in range(args.num_layers):
    features = model(X, i)
    print(i, features[0].shape)
    pixel_resolution.append(features.shape[2:])


def get_model_weights(model):
    p = [list(model.parameters())[i].data.cpu().numpy() for i in range(len(list(model.parameters())))]
    return p

params = get_model_weights(model)

weights = params[::2]

print('counting number of operation in one model')
operation_by_layer = []
for i, (WH, W) in enumerate(zip(pixel_resolution, weights)):
    print(i, np.prod(WH), W.shape)
    mult_operation = np.prod(WH)*np.prod(W.shape)
    add_operation = np.prod(WH)*np.prod(W.shape) #with biases
    overall_operation = mult_operation + add_operation
    operation_by_layer.append(overall_operation)
    print(i, overall_operation)

print('counting number of operations in WA ensemblings')
single_model_operations = np.sum(operation_by_layer)
ind_ensembling_operations = single_model_operations*args.num_models

WA_ensembling_operations = []
for i in range(1, args.num_layers):
    WA_operations = np.sum(operation_by_layer[:i])+np.sum(operation_by_layer[i:])*args.num_models
    print(i, WA_operations)
    WA_ensembling_operations.append(WA_operations)

WA_ensembling_operations_ratio = np.array(WA_ensembling_operations)/ind_ensembling_operations

print('WA_ensembling_operations', WA_ensembling_operations)
print('WA_ensembling_operations_ratio', WA_ensembling_operations_ratio)


np.savez(
    os.path.join(args.dir, 'number_operations.npz'),
    WA_ensembling_operations=WA_ensembling_operations,
    WA_ensembling_operations_ratio=WA_ensembling_operations_ratio,
    single_model_operations_by_layer=operation_by_layer,
    ind_ensembling_operations=ind_ensembling_operations,
)

plt.figure(0)
plt.scatter(np.arange(1, args.num_layers+1), [1]+list(WA_ensembling_operations_ratio))
plt.plot(np.arange(1, args.num_layers+1), [1/args.num_models]*args.num_layers, c='g')
plt.legend('WA ensembling operations ratio to Ind ensembling')
plt.xlabel('layer to perform WA on')
plt.ylabel('ratio')
plt.savefig(os.path.join(args.dir, 'wa_ensembling_operations_ratio.png'))

plt.figure(1)
plt.scatter(np.arange(1, args.num_layers+1), operation_by_layer)
plt.legend('single_model operations by layer')
plt.xlabel('layer')
plt.ylabel('number of operations')
plt.savefig(os.path.join(args.dir, 'single_model_operations.png'))




