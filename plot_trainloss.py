import numpy as np
import matplotlib.pyplot as plt
data_path='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/6/'
data_path1='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/train_normal_10/train_record_epoch_lr_trainloss_trainaccur_testnll_testaccurPreResNet110.npz'
data_path2='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/8/test_record_PreResNet110.npz'
data_path3='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/train_normal_7/train_record_epoch_lr_trainloss_trainaccur_testnll_testaccurPreResNet110.npz'
x1=np.load(data_path1)
x2=np.load(data_path2)
x3=np.load(data_path3)
# data1=x1['train_and_value'][:,2].tolist()
data1=x1['train_and_value'][:,5].tolist()
data2=x2['record_train']
data2=[data2[i]['accuracy'] for i in range(201)]
data21=data2[2:52]
data22=[data2[i]+1 for i in range(149,199)]
data2=data2[2:102]
data12=data1+data21+data22

# data3=x3['train_and_value'][:,2]
data3=x3['train_and_value'][:,5]

plt.plot(range(300), data12, color='red', label='SBE')
plt.plot(range(300), data3, color='blue', label='standard learning rate')
plt.xlabel('Epochs')
plt.ylabel('Testing accuracy')
# plt.title('')
plt.grid()
plt.legend()
plt.savefig(data_path+'sbecomparestandard1.pdf',dpi=400)
plt.show()