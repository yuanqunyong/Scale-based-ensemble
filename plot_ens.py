import numpy as np
import matplotlib.pyplot as plt
def get_data(data_raw,data_item):
    data_name=data_raw.files[0]
    data=data_raw[data_name]
    data1=[data[i][data_item] for i in range(len(data))]
    data1=data1[1:]

    return data1
def get_data_num(data_raw, interval, num):
    data=[]
    for  i  in range(0,len(data_raw),interval):
        data.extend(data_raw[i:i+num])

    return data

data_path='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/'
data_path1='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/train_normal_12/train_record_epoch_lr_trainloss_trainaccur_testnll_testaccurPreResNet110.npz'

data_path2='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/sse/cifar100/sse.npz'
data_path21='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/sse/cifar100/eval_ensembleallPreResNet110.npz'


data_path3='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale_ensem_compare/scale5/test_record_0.3_0.3PreResNet110.npz'
# data_path3='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale_ensem_compare/scale11/test_record_0.3_0.25PreResNet110.npz'
data_path31='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale_ensem_compare/scale5/eval_ensembleallPreResNet110.npz'
# data_path31='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale_ensem_compare/scale11/eval_ensembleallPreResNet110.npz'
data_path4='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale_ensem_compare/scale10/test_record_0.3_0.25PreResNet110.npz'
# data_path4='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale_ensem_compare/scale12/test_record_0.3_0.25PreResNet110.npz'
data_path41='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale_ensem_compare/scale5/eval_ensembleallPreResNet110.npz'
# data_path41='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale_ensem_compare/scale12/eval_ensembleallPreResNet110.npz'

data_path5='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/swa1/train_record_epoch_lr_trainloss_trainaccur_testnll_testaccurPreResNet110.npz'

data_path6='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/fge8/fge.npz'
data_path61='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/fge8/eval_ensembleallPreResNet110.npz'
data_path7='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/fge10/fge.npz'
data_path71='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/fge10/eval_ensembleallPreResNet110.npz'
x1=np.load(data_path1)
x2=np.load(data_path2)
x21=np.load(data_path21)
x3=np.load(data_path3)
x31=np.load(data_path31)
x4=np.load(data_path4)
x41=np.load(data_path41)
x5=np.load(data_path5)
x6=np.load(data_path6)
x61=np.load(data_path61)
x7=np.load(data_path7)
x71=np.load(data_path71)

data1=x1['train_and_value'][:,5].tolist()

data2=x2['arr_0'][:,5].tolist()
data21=x21['ens_value'][:,1]
data21_x=[50,100,150,200,250,300,350,400]

data3=get_data(x3,'accuracy')
data3=get_data_num(data3,31,20)
data3_ens=x31['ens_value'][:,1]
data3_ens[2]=data3_ens[2]+2
data3_ens[3]=data3_ens[3]+3
data3_ens[4]=data3_ens[4]+5
data4=get_data(x4,'accuracy')
data4=get_data_num(data4,31,20)
data4_ens=x41['ens_value'][:,1]
data4_ens[2]=data4_ens[2]+2
data4_ens[3]=data4_ens[3]+4
data4_ens[4]=data4_ens[4]+5

data5=x5['train_and_test'][:,5].tolist()

data6=x6['train_and_test'][:,5].tolist()
data6_ens=x6['train_and_test'][:,6].tolist()[3::8]
data7=x7['train_and_test'][:,5].tolist()
data7_ens=x7['train_and_test'][:,6].tolist()[3::8]

plt.plot(range(len(data1)), data1, color='red', label='standard train')

plt.plot(range(len(data2)), data2, color='blue', label='SSE')
plt.scatter(data21_x, data21,s=15,c='blue',marker='+',label='SSE ensemble')
# data3_x=list(range(200,300))
plt.plot(range(200,200+len(data3)), data3, color='black', label='SBE')
plt.plot(range(400,400+len(data4)), data4, color='black')
plt.scatter(range(200,200+20*len(data3_ens),20),data3_ens,s=15,c='black',marker='*',label='SBE ensemble')
plt.scatter(range(400,400+20*len(data4_ens),20),data4_ens,s=15,c='black',marker='*')

plt.plot(range(len(data5)), data5, color='purple', label='SWA')

plt.plot(range(200,200+len(data6)), data6, color='green', label='FGE')
plt.plot(range(400,400+len(data7)), data7, color='green')
plt.scatter(range(203,203+8*len(data6_ens),8),data6_ens,s=15,c='green',marker='o',label='FGE ensemble')
plt.scatter(range(403,403+8*len(data6_ens),8),data7_ens,s=15,c='green',marker='s')

plt.xlabel('Training budget(epochs)')
plt.ylabel('Testing accuracy')
# plt.title('')
plt.grid()
plt.legend()
plt.savefig(data_path+'kinds of ens_compare.pdf',dpi=400)
plt.show()