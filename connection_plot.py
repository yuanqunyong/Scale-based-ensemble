import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
file_path_save='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/connect/'
#
# file_path1='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale/connect/1/chain0-8.npz'
# file1 = np.load(file_path1)
# file_path2='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale/connect/2/chain0-8.npz'
# file2 = np.load(file_path2)
# file_path3='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale/connect/3/chain0-8.npz'
# file3 = np.load(file_path3)
# file_path4='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale/connect/4/chain0-8.npz'
# file4 = np.load(file_path4)
# file_path5='/home/robot/yuanqunyong/dnn-mode-connectivity-master/train/preresnet/scale/connect/5/chain0-8.npz'
# file5 = np.load(file_path5)

file_path1='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/connect/norma2/chain0-8.npz'
file1 = np.load(file_path1)
file_path2='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/connect/norma3/chain0-8.npz'
file2 = np.load(file_path2)
file_path3='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/connect/norma4/chain0-8.npz'
file3 = np.load(file_path3)
file_path4='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/connect/norma5/chain0-8.npz'
file4 = np.load(file_path4)
file_path5='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/scale/connect/norma6/chain0-8.npz'
file5 = np.load(file_path5)
y1=file1['te_err']
y2=file2['te_err']
y3=file3['te_err']
y4=file4['te_err']
y5=file5['te_err']
x=np.linspace(0,1,num=50,endpoint=True)
# x=range(1,81)
# y=file['tr_nll']
plt.plot(np.linspace(0,1,num=len(y1),endpoint=True), y1,color='red',linewidth=1,linestyle='-' ,label='with 1-th')
plt.plot(np.linspace(0,1,num=len(y2),endpoint=True), y2,color='blue',linewidth=1,linestyle='-',label='with 2-th')
plt.plot(np.linspace(0,1,num=len(y3),endpoint=True), y3,color='purple',linewidth=1,linestyle='-',label='with 3-th')
plt.plot(np.linspace(0,1,num=len(y4),endpoint=True), y4,color='green',linewidth=1,linestyle='-',label='with 4-th')
plt.plot(np.linspace(0,1,num=len(y5),endpoint=True), y5,color='black',linewidth=1,linestyle='-',label='with 5-th')
plt.title('Ind')
plt.ylabel('testing error')
plt.xlabel('t')
plt.grid()
plt.legend()

# plt.savefig(file_path_save+'train_nll.pdf', format='pdf', bbox_inches='tight',dpi=400)
plt.savefig(file_path_save+'Ind_test_err.pdf', format='pdf', bbox_inches='tight',dpi=600)
plt.show()
# for var in file.files:
#     print(var)
#     print(len(file[var]))
#
