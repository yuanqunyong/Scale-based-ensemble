
import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

import data
import models
import utils
import pickle
from correlation_analyze import  correlation_computing,correlation_computing_for_modelselect
from itertools import combinations

parser = argparse.ArgumentParser(description='Ensemble evaluation')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--use_ckptpath', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--use_ens_opt', action='store_true',
                    help='switches computing ens_opt (default: validation)')
parser.add_argument('--transform', type=str, default='ResNet', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='/home/robot/yuanqunyong/PyTorch-AutoNEB-master/tmp', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--ensemble_num_list', type=str, action='append', metavar='N',
                    help='ensemble_num_for_optim (default: 2)')
parser.add_argument('--ensemble_num_need', type=int, metavar='N',
                    help='ensemble_num_need (default: 10)')
parser.add_argument('--ensemble_num_final', type=str, action='append', metavar='N',
                    help='ensemble_num_final (default: 6)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='PreResNet110', metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, action='append', metavar='CKPT',
                    help='checkpoint to eval, pass all the models through this parameter')
parser.add_argument('--dir', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/train/vggbn', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')
parser.add_argument('--dir_ckpt', type=str, default='/home/robot/yuanqunyong/adaptive-momentum-master/train/vggbn', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')


args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'ensemble.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
filePath =args.dir_ckpt
if args.use_ckptpath==True:
    ckpt_name=os.listdir(filePath)
    ckpt_name=[ckpt_name[i] for i in range(len(ckpt_name)) if 'pt'in ckpt_name[i]]

    ckpt_path=[os.path.join(args.dir, ckpt_name[i]) for i in range(len(ckpt_name))]
else:
    ckpt_path=args.ckpt
torch.backends.cudnn.benchmark = True
ckpt_path=sorted(ckpt_path)
print(ckpt_path)
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
model_second = architecture.base(num_classes=num_classes, **architecture.kwargs)

criterion = F.cross_entropy

model.cuda()
param_dist_list=[]

def compute_dist(model1,model2):

    global param_dist_list

    d = 0.
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1 = param1.data.cpu().numpy()
        param2 = param2.data.cpu().numpy()
        d += np.sum((param1 - param2) ** 2)
    print('param_dist is %s'%d)
    param_dist_list.append(np.sqrt(d))

    with open(args.dir + "/param_dist_list.pkl", "wb") as f:
        pickle.dump(param_dist_list, f)

def test_accuray_ens(model,ckpt_list,loaders):
    acc_dir=dict()
    acc_list=[]
    predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

    for i,path in enumerate(ckpt_list):
        # print(path)
        temp=np.zeros((2))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])

        predictions, targets = utils.predictions(loaders['test'], model)
        acc = 100.0 * np.mean(np.argmax(predictions, axis=1) == targets)

        predictions_sum += predictions
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)
        temp[0]=acc
        temp[1]=ens_acc
        acc_list.append(temp)
        acc_dir[i]=temp
        print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))

    return acc_dir

def test_accuray_save(model,ckpt_list,loaders,name):
    acc_dir=dict()
    acc_list=[]
    predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

    for i,path in enumerate(ckpt_list):
        print(path)
        temp=np.zeros((2))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])

        predictions, targets = utils.predictions(loaders['test'], model)
        acc = 100.0 * np.mean(np.argmax(predictions, axis=1) == targets)

        predictions_sum += predictions
        ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)
        temp[0]=acc
        temp[1]=ens_acc
        acc_list.append(temp)
        acc_dir[i]=temp
        print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))
    acc_value=np.vstack(acc_list)
    np.savez(
            os.path.join(args.dir, 'eval_ensemble'+name+args.model+'.npz'),
            ens_value=acc_value)
    return acc_dir
#computing optim ensembles value according to given ckpt_num
if args.use_ens_opt==True:

    for ens_num  in args.ensemble_num_list:
        comb=list(combinations(range(len(ckpt_path)), int(ens_num)))
        ens_opt=[0,0]
        for  kk  in comb:
            ckpt_temp=[ckpt_path[i] for i in kk]
            ens_val_temp=test_accuray_ens(model, ckpt_temp,loaders)
            val_temp=ens_val_temp[len(ens_val_temp)-1][1]
            if val_temp> ens_opt[0]:
                ens_opt[0]=val_temp
                ens_opt[1]=ckpt_temp

        print(ens_opt)
        np.savez(
            os.path.join(args.dir, 'eval_ensemble' +str(ens_num)+ args.model + '.npz'),
            ens_value=np.asarray(ens_opt[0]),ckptlist=np.asarray(ens_opt[1]))



else:
    # ours model selection method
    acc_dict = test_accuray_ens(model, ckpt_path, loaders)
    acc_list_sorted = sorted(acc_dict.items(), key=lambda item:item[1][0], reverse=True)
    print(acc_list_sorted)


    ensemble_num_need= args.ensemble_num_need
    ensemble_num_final=[int(i) for i in args.ensemble_num_final]
    ensemble_num_final= sorted(ensemble_num_final)
    # for  item1   in args.ensemble_num_need:

        # ensemble_num_need = item1

    ckpt_list_index_and_acc = acc_list_sorted[:ensemble_num_need]
    ckpt_list = [ckpt_path[item[0]] for item in ckpt_list_index_and_acc]

    print(ckpt_list)

    ckpt_listindex_final=[0]
    ckpt_listindex=list(range(len(ckpt_list)))
    ckpt_listindex.pop(0)

    for i in range(ensemble_num_final[-1]-1):

            insert_item=[0, 0]
            for j in  ckpt_listindex :
                ckpt_list_final=[ckpt_list[n] for n in ckpt_listindex_final]
                # temp, _=correlation_computing_for_modelselect(model,ckpt_list[j],*ckpt_list_final,loaders=loaders,num_classes=num_classes)
                _, accuracy=correlation_computing_for_modelselect(model,ckpt_list[j],*ckpt_list_final,loaders=loaders,num_classes=num_classes)

                # if temp<insert_item[1]:
                if  accuracy>insert_item[1]:
                    insert_item[0]=j
                    # insert_item[1]=temp
                    insert_item[1]=accuracy
            ckpt_listindex_final.append(insert_item[0])
            ckpt_listindex.remove(insert_item[0])

            if i+2 in ensemble_num_final:

                ckpt_final = [ckpt_list[item] for item in ckpt_listindex_final]
                print(ckpt_final)

                # test_accuray_ens_vote(model, ckpt_final,loaders)

                test_accuray_save(model,ckpt_final,loaders,str(ensemble_num_need)+'-'+str(i+1))








