import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import data
import models
import utils

parser = argparse.ArgumentParser(description='FGE training')

parser.add_argument('--dir', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/tmp2/sse/sgd3', metavar='DIR',
                    help='training directory (default: /tmp/sse)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='ResNet', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='/home/robot/yuanqunyong/PyTorch-AutoNEB-master/tmp', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='PreResNet110', metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/tmp/curve_sgd3/checkpoint_ealy-1500.pt', metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--cycles', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--initial_lr', type=float, default=0.1, metavar='LR1',
                    help='initial learning rate (default: 0.05)')
# parser.add_argument('--lr_2', type=float, default=0.0001, metavar='LR2',
#                     help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'sse.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

# checkpoint = torch.load(args.ckpt)
# start_epoch = checkpoint['epoch'] + 1
# model.load_state_dict(checkpoint['model_state'])
model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=args.momentum,
    weight_decay=args.wd
)
# optimizer.load_state_dict(checkpoint['optimizer_state'])

# ensemble_size = 0
predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'ens_acc', 'time']

values_list=[]
epochs_per_cycle = args.epochs // args.cycles
for i in range(args.cycles):
    time_ep = time.time()
    for j in range(epochs_per_cycle):

        lr = utils.proposed_lr(args.initial_lr, j, epochs_per_cycle)
        optimizer.param_groups[0]["lr"] = lr
        train_res = utils.train_se(loaders['train'], model, optimizer, criterion)
        test_res = utils.test(loaders['test'], model, criterion)
        time_ep = time.time() - time_ep
        values = [i, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
                  test_res['accuracy'], time_ep]
        values_list.append(values)
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    predictions, targets = utils.predictions(loaders['test'], model)
    predictions_sum += predictions
    ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)
    print('ens_acc is {}'.format(ens_acc))


    utils.save_checkpoint(
            args.dir,
             i*epochs_per_cycle,
            name='sse',
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )



np.savez(
    os.path.join(args.dir, 'sse.npz'),
    train_and_test=values_list
)
