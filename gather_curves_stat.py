import os
from tqdm import tqdm
import subprocess
import torch
import numpy as np
import argparse
import sys
from utils.train_utils import get_models_path

parser = argparse.ArgumentParser(description='gathering curve statistics')
parser.add_argument('--dir', type=str, default='/home/ivan/dnn-mode-connectivity/curves/LinearOneLayer100/', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
parser.add_argument('--model', type=str, default='LinearOneLayer100', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default='PolyChain', metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--name', type=str, default='400')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--device', type=int, default=1)

parser.add_argument('--point_finder', type=str, default='gd', help='PointFinder', metavar='POINTFINDER')
parser.add_argument('--method', type=str, default=None, help='method to apply in PointFinder', metavar='METHOD')
parser.add_argument('--end_time', type=int, default=1, help='time then path reach the --end checkpoint', metavar='ENDTIME')


args = parser.parse_args()

path = args.dir
name = args.name
architecture = args.model
num_bends = args.num_bends
curve = args.curve
epochs = args.epochs
num_points = args.num_points
device = args.device
device_id = 'cuda:' + str(device)

torch.cuda.set_device(device_id)

file_paths = get_models_path(path, name)
print('file_paths', file_paths)

print('finding curves')
tr_acc = []
te_acc = []
for i in tqdm(range(len(file_paths) - 1)):
    curve1 = file_paths[i].split('/')[-2]
    curve2 = file_paths[i + 1].split('/')[-2]
    full_name = '{0}_{1}_{2}_{3}_{4}_{5}_nb{6}'.format(architecture, name, curve1, curve2,
                                                 args.point_finder, args.method, args.num_bends)

    connect_dir = 'experiments/connect/' + full_name

    if args.point_finder == 'gd':
        print('connecting checkpoints', connect_dir)
        bashCommand = ('python train.py --dir={0} ' + \
                      '--model={1} --data_path=data --epochs={2} --curve={3}  --num_bends={4} --fix_start --fix_end --cuda ' + \
                      '--init_start={5} ' + \
                      '--init_end={6} --device={7} --dataset={8}').format(
            connect_dir, architecture, epochs, curve, num_bends, file_paths[i], file_paths[i+1], args.device, args.dataset)
        print('bash:', bashCommand)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(error)

    eval_dir = 'experiments/eval/' + full_name
    print('eval curve', eval_dir)
    bashCommand = ('python eval_curve.py --dir={0} --model={1} --cuda ' +\
    '--data_path=data --curve={2}  ' +\
    '--ckpt={3} --num_points={4} --point_finder={5} --method={6} --end_time={7} --start={8} --end={9} --device={10} ' +\
    '--num_bends={11} --dataset={12}' ).format(
         eval_dir, architecture, curve,
         connect_dir+'/checkpoint-{}.pt'.format(epochs),
         num_points, args.point_finder, args.method,
          args.end_time, file_paths[i], file_paths[i+1], args.device, num_bends, args.dataset,
         )
    print('bash:', bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(error)

    stat_dir = eval_dir + '/curve.npz'
    print('taking stat for the one curve', stat_dir)
    stat = np.load(stat_dir)
    tr_acc.append(100 - stat['tr_err_max'])
    te_acc.append(100 - stat['te_err_max'])

min_acc_dir = 'experiments/accuracy/{}_{}_{}_{}_ep{}_{}_{}'.format(architecture, name, curve, num_bends, epochs,
                                                                   args.point_finder, args.method)
print('saving accuracy stat', min_acc_dir)

def get_mean_svd(stat):
    train = np.array(stat['train'])
    test = np.array(stat['test'])
    return train.mean(), train.std(), test.mean(), test.std()


print('tr_acc', tr_acc)
print('te_acc', te_acc)

stat = {'train': tr_acc, 'test': te_acc}
print('train, test mean and std', get_mean_svd(stat))
tr_mean, tr_std, te_mean, te_std = get_mean_svd(stat)

os.makedirs(min_acc_dir, exist_ok=True)
np.savez(
    os.path.join(min_acc_dir, 'acc_stat.npz'),
    tr_acc=tr_acc,
    te_acc=te_acc,
    tr_mean=tr_mean,
    tr_std=tr_std,
    te_mean=te_mean,
    te_std=te_std,
)

with open(os.path.join(min_acc_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')






