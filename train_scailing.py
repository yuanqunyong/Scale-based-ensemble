import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils
import numpy as np
from torchsummary import summary


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/tmp2/train_scale/sgd/wide28*10_1', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='WideResNet28x10', metavar='MODEL',
                    help='model name (default: None)')
# parser.add_argument('--model', type=str, default='WideResNet28x10', metavar='MODEL', required=True,
#                     help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                    help='fix end point (default: off)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--save_model_per_iter', dest='save_model_per_iter', action='store_false',
                    help='save model per_iter')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--scaling_checkpoint', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/tmp/curve_sgd3/checkpoint_ealy-1500.pt', metavar='CKPT',
                    help='checkpoint to scaling training from (default: None)')
# parser.add_argument('--scaling_checkpoint', type=str, default='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train_normal/wideresnet/wide_res_lastepoch-200.pt', metavar='CKPT',
#                     help='checkpoint to scaling training from (default: None)')


parser.add_argument('--epochs', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')

parser.add_argument('--cycle', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lr_1', type=float, default=0.05, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lr_2', type=float, default=0.0005, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
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

if args.curve is None:
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
else:
    curve = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        args.fix_start,
        args.fix_end,
        architecture_kwargs=architecture.kwargs,
    )
    base_model = None

model.cuda()


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    else:
        factor = 0.01
    return factor * base_lr


criterion = F.cross_entropy
regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd if args.curve is None else 0.0
)


# optimizer = torch.optim.Adam(
#     filter(lambda param: param.requires_grad, model.parameters()),
#     lr=1e-3,
#     weight_decay=args.wd if args.curve is None else 0.0)
#
if args.scaling_checkpoint is not None:
    print('scaling taining from %s' % args.scaling_checkpoint)
    checkpoint = torch.load(args.scaling_checkpoint)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])


test_res_before_scale= utils.test(loaders['test'], model, criterion, regularizer)
print('test_res_before_scale is {}'.format(test_res_before_scale['accuracy']))

def unit_conv(conv_temp, conv_shape,scale_value,mean):
    for i in range(conv_shape[0]):
        tr=torch.norm(conv_temp[i]).data.cpu().numpy()
        if  tr> scale_value :
            # print(tr,mean/tr)
            conv_temp[i]=conv_temp[i]*torch.from_numpy(np.asarray([0.01],dtype='float32')).cuda()
        conv_temp=conv_temp.reshape(conv_shape)

    # return  conv_temp

    # conv2_list=[]
def scaling(model,scale_value,mean):
    for layer in model.named_modules():
        # print(layer)
        if  'conv2' in layer[0]:
            conv_shape = layer[1].weight.shape
            conv_temp = layer[1].weight.detach().reshape(conv_shape[0],-1)
            conv_temp = unit_conv(conv_temp, conv_shape, scale_value,mean)
            layer[1].weight.data.copy_(conv_temp)




#computing scaling coefficient

# num_need_adj=round(length/10)


# for scale_value in [tr[round(length/20)], tr[round(length/19)],tr[round(length/18)],tr[round(length/17)]]:
for scale_value in range(5):
    # checkpoint = torch.load(args.scaling_checkpoint)
    # model.load_state_dict(checkpoint['model_state'])
    # optimizer.load_state_dict(checkpoint['optimizer_state'])
    tr = []
    for layer in model.named_modules():
        if 'conv2' in layer[0]:
            conv_shape = layer[1].weight.shape
            conv_temp = layer[1].weight.detach().reshape(conv_shape[0], -1)

            for i in range(conv_shape[0]):
                tr.append(torch.norm(conv_temp[i]).data.cpu().numpy())
    tr.sort()
    length = len(tr)
    tr_np = np.asarray(tr)
    value_mean = np.mean(tr_np)
    value_median=np.median(tr_np)
    # print(tr)
    # print(length)
    print(value_mean,value_median)
    # for val in tr:
    #     print(val)
    # print(tr[length-50])

    scaling(model,tr[length-60],value_mean)
    test_res_after_scale= utils.test(loaders['test'], model, criterion, regularizer)
    print('test_res_after_scale is {}'.format(test_res_after_scale['accuracy']))

    for epoch in range(0, 4):
        lr = learning_rate_schedule(args.lr, epoch, args.epochs)
        utils.adjust_learning_rate(optimizer, lr)
        train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
        # lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
        test_res_after_scale_and_train= utils.test(loaders['test'], model, criterion, regularizer)
        print('test_res_after_scale_and_train is {}'.format(test_res_after_scale_and_train['accuracy']))

    start_epoch=1500

    utils.save_checkpoint_saling(
            args.dir, start_epoch+2,
            str(scale_value),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

