import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F
import os
import curves
import data
import models
import utils
import  random
import numpy as np
from torchsummary import summary


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/train/vggbn/scale/cycling_style', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--dir', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/train/vggbn/scale/cycling_style', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--rand', action='store_true',
                    help='switches between define and random')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='WideResNet28x10', metavar='MODEL', required=True,
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
parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT', required=True,
                    help='checkpoint to eval, pass all the models through this parameter')


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
summary(model, (3, 32, 32))

def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    else:
        factor = 0.1
    return factor * base_lr


criterion = F.cross_entropy
regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
optimizer = torch.optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd if args.curve is None else 0.0
)

checkpoint = torch.load(args.ckpt[0])
num_scale=10
# for i, ckp in enumerate(args.ckpt):
for i in range(num_scale):
    # checkpoint = torch.load(ckp)
    key_weight_name=[]
    key_bias_name=[]
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    test_res_before_scale = utils.test(loaders['test'], model, criterion, regularizer)
    print('test_res_before_scale is {}'.format(test_res_before_scale['accuracy']))
    scale_list=[1/5,1/4,1/3,1/2,2,3,4]
    if args.model=='VGG19BN'or args.model=='VGG16BN'or args.model == 'WideResNet28x10' or args.model == 'WideResNet16x4':
        if args.model=='VGG19BN'or args.model=='VGG16BN':

            for key in checkpoint['model_state'].keys():
                if  'layer'in key and  'weight' in key:
                    key_weight_name.append(key)
            for key in checkpoint['model_state'].keys():
                if  'layer'in key and  'bias' in key:
                    key_bias_name.append(key)
        else:
            for key in checkpoint['model_state'].keys():
                if 'conv1.weight' in key:
                    key_weight_name.append(key)
            for key in checkpoint['model_state'].keys():
                if 'conv1.bias' in key:
                    key_bias_name.append(key)

        if  args.rand==True:

            indexchoice_key_name=random.sample(range(len(key_weight_name)), int(len(key_weight_name)/5))

            for  name_index in indexchoice_key_name:
                scale_value=random.sample(scale_list,1)
                checkpoint['model_state'][key_weight_name[name_index]].data.copy_(torch.mul(checkpoint['model_state'][key_weight_name[name_index]].detach(), scale_value[0]))
                checkpoint['model_state'][key_bias_name[name_index]].data.copy_(torch.mul(checkpoint['model_state'][key_bias_name[name_index]].detach(), scale_value[0]))
        else:
            for name_weight, name_bias in zip(key_weight_name[len(key_weight_name)//2:], key_bias_name[len(key_weight_name)//2:]):
                scale_value = random.sample(scale_list, 1)
                checkpoint['model_state'][name_weight].data.copy_(torch.mul(checkpoint['model_state'][name_weight].detach(), scale_value[0]))

                checkpoint['model_state'][name_bias].data.copy_(torch.mul(checkpoint['model_state'][name_bias].detach(), scale_value[0]))

        model.load_state_dict(checkpoint['model_state'])

    elif args.model == 'PreResNet110'or args.model == 'PreResNet164':

        for key in checkpoint['model_state'].keys():
            if 'conv1'in key or 'conv2' in key:
                key_weight_name.append(key)
        if args.rand == True:
            indexchoice_key_name = random.sample(range(len(key_weight_name)), int(len(key_weight_name) / 3))
            for name_index in indexchoice_key_name:
                scale_value = random.sample(scale_list, 1)
                checkpoint['model_state'][key_weight_name[name_index]].data.copy_(
                    torch.mul(checkpoint['model_state'][key_weight_name[name_index]].detach(), scale_value[0]))
        else:
            for name_weight in key_weight_name[len(key_weight_name)//2:]:
                scale_value = random.sample(scale_list, 1)
                checkpoint['model_state'][name_weight].data.copy_(
                        torch.mul(checkpoint['model_state'][name_weight].detach(), scale_value[0]))
        model.load_state_dict(checkpoint['model_state'])
    elif args.model == 'DenseNet121' or args.model == 'DenseNet169':

        for key in checkpoint['model_state'].keys():
            if 'conv1.weight' in key and 'dense' in key:
                key_weight_name.append(key)

        print(key_weight_name)
        if  args.rand==True:

            indexchoice_key_name=random.sample(range(len(key_weight_name)), int(len(key_weight_name)/4))

            for  name_index in indexchoice_key_name:
                scale_value=random.sample(scale_list,1)
                checkpoint['model_state'][key_weight_name[name_index]].data.copy_(torch.mul(checkpoint['model_state'][key_weight_name[name_index]].detach(), scale_value[0]))

        else:
            for name_weight in key_weight_name[len(key_weight_name)//4:]:
                scale_value = random.sample(scale_list, 1)
                checkpoint['model_state'][name_weight].data.copy_(
                        torch.mul(checkpoint['model_state'][name_weight].detach(), scale_value[0]))
        model.load_state_dict(checkpoint['model_state'])

    for epoch in range(0, 25):
        if epoch==0:
            lr = learning_rate_schedule(args.lr, epoch, args.epochs)
            utils.adjust_learning_rate(optimizer, lr)

            train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
            # lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
            test_res_after_scale_and_train= utils.test(loaders['test'], model, criterion, regularizer)
            record=test_res_after_scale_and_train['accuracy']
            print('test_res_after_scale_and_train is {}'.format(test_res_after_scale_and_train['accuracy']))
        #
            start_epoch=200

            utils.save_checkpoint_saling(
                    args.dir, start_epoch+2,
                    str(i),
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )
        else:
            lr = learning_rate_schedule(args.lr, epoch, args.epochs)
            utils.adjust_learning_rate(optimizer, lr)

            train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
            # lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
            test_res_after_scale_and_train = utils.test(loaders['test'], model, criterion, regularizer)
            print('test_res_after_scale_and_train is {}'.format(test_res_after_scale_and_train['accuracy']))
            if test_res_after_scale_and_train['accuracy']>=record:
                record = test_res_after_scale_and_train['accuracy']
                utils.save_checkpoint_saling(
                    args.dir, start_epoch + 2,
                    str(i),
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )


