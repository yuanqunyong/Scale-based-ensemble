import pickle
import matplotlib.pyplot as plt

save_path='/home/robot/yuanqunyong/adaptive-momentum-master/picture'
data_path2='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/sse/param_dist_list.pkl'
data_path3='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/fge2/param_dist_list.pkl'
data_path4='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/1/param_dist_list.pkl'
data_path1='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/ind/param_dist_list.pkl'
fr1 = open(data_path1,'rb')
fr2 = open(data_path2,'rb')
fr3 = open(data_path3,'rb')
fr4 = open(data_path4,'rb')
data1 = pickle.load(fr1)
data2 = pickle.load(fr2)
data2.append(114.3)
data3 = pickle.load(fr3)
data3=data3[:5]
data3.insert(1,data3[4])
del data3[-1]
data3=data3[::-1]
data4 = pickle.load(fr4)
data4=data4[::-1]
data4.append(171.43)

plt.plot(range(1,6), data1[:5], color='black', label='Ind' ,marker='D')
plt.plot(range(1,6), data2[:5], color='blue', label='SSE',marker='o')
plt.plot(range(1,6), data3[:5], color='green', label='FGE',marker='p')
plt.plot(range(1,6), data4[:5], color='red', label='SBE',marker='s')
plt.xlabel('The $i$ th network model')
plt.ylabel('Parameter Distance')
plt.title('PreResNet110+CIFAR100')
plt.grid()
plt.legend()
plt.savefig(save_path+'preresnetpredistancecifar100.pdf',dpi=400)
plt.show()