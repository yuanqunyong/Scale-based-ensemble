import numpy as np
import os
import torch
import torch.nn.functional as F
from math import pi
from math import cos

import curves



def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, iter, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    # filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    filepath = os.path.join(dir, './epoch_{}.batch_{}.pt'.format(epoch, iter))
    torch.save(state, filepath)

def save_checkpoint_1(dir, epoch, iter, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    # filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    filepath = os.path.join(dir, './epoch_{}.batch_{}.pt'.format(epoch, iter))
    torch.save(state, filepath)



def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)

def save_checkpoint_saling(dir, epoch, name, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    # filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    filepath = os.path.join(dir, './scaling_{}.pt'.format(str(name)))
    torch.save(state, filepath)

def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2
# def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None, save_model_per_iter=False,epoch=1):
# def train(train_loader, model, optimizer, criterion, args,regularizer=None, lr_schedule=None, save_model_per_iter=False,epoch=1):
#     loss_sum = 0.0
#     correct = 0.0
#
#     num_iters = len(train_loader)
#     model.train()
#     for iter, (input, target) in enumerate(train_loader):
#         if lr_schedule is not None:
#             lr = lr_schedule(iter / num_iters)
#             adjust_learning_rate(optimizer, lr)
#         input = input.cuda(async=True)
#         target = target.cuda(async=True)
#
#         output = model(input)
#         loss = criterion(output, target)
#         if regularizer is not None:
#             loss += regularizer(model)
#
#         if (save_model_per_iter is True) &(iter<=50)& ((epoch == 1) | (epoch == 10) | (epoch == 50) | (epoch == 100)| (epoch == 200)| (epoch == 400)):
#             print('Saving intermediate model..')
#             utils.save_checkpoint(
#                 args.dir,
#                 epoch,
#                 iter,
#                 model_state=model.state_dict(),
#                 optimizer_state=optimizer.state_dict()
#             )
#             # state = {
#             #     'net': model,
#             #     'iter': iter,
#             # }
#             # with open('./epoch_{}.batch_{}.pt'.format(epoch, iter), 'wb') as f:
#             #     torch.save(state, f)
#
#
#         optimizer.zero_grad()
#         loss.backward()
#
#         optimizer.step()
#
#         loss_sum += loss.item() * input.size(0)
#         pred = output.data.argmax(1, keepdim=True)
#         correct += pred.eq(target.data.view_as(pred)).sum().item()
#
#     return {
#         'loss': loss_sum / len(train_loader.dataset),
#         'accuracy': correct * 100.0 / len(train_loader.dataset),
#     }

def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        # optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def train_se(train_loader, model, optimizer, criterion, regularizer=None):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    for iter, (input, target) in enumerate(train_loader):

        input = input.cuda(async=True)
        target = target.cuda(async=True)
        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }

def train_1(epoch):
    global trainloader
    global optimizer
    global args
    global model
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_list = []
    for lr_ in lr_sch:
        if epoch <= lr_[0]:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * lr_[1],
                                        momentum=0,
                                        weight_decay=args.wdecay)
            break

    optimizer.zero_grad()
    if not hasattr(train, 'nb_samples_seen'):
        train.nb_samples_seen = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if train.nb_samples_seen + args.mbs== args.bs:
            optimizer.step()

            optimizer.zero_grad()
            if (args.save_model_per_iter is True)&((epoch==1)|(epoch==10)|(epoch==25)|(epoch==100)):
                print('Saving intermediate model..')
                state = {
                    'net': model,
                    'iter': batch_idx,
                }
                with open(args.save_dir + '/epoch_{}.batch_{}.pt'.format(epoch, batch_idx), 'wb') as f:
                    torch.save(state, f)
            train.nb_samples_seen = 0
        else:
            train.nb_samples_seen += args.mbs
        loss_list.append(loss.data[0])
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if not args.cluster:
            progress_bar(batch_idx, len(trainloader), 'Epoch {:3d} | Loss: {:3f} | Acc: {:3f}'
                         .format(epoch, train_loss / (batch_idx + 1), 100. * correct / total))
    print('Saving model..')
    state = {
        'net': model,
        'iter': epoch,
    }
    with open(args.save_dir + '/epoch_{}.pt'.format(epoch), 'wb') as f:
        torch.save(state, f)

    return sum(loss_list) / float(len(loss_list)), 100. * correct / total


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()

    for input, target in test_loader:
        input = input.cuda(async=True)
        target = target.cuda(async=True)

        output = model(input, **kwargs)
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(async=True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda(async=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
