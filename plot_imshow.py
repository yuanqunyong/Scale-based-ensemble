import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as col
# sphinx_gallery_thumbnail_number = 2
data_path='/home/robot/yuanqunyong/adaptive-momentum-master/train/preresnet/ind/correlation_matrixPreResNet110.npz'
# method_type='SBE'
# method_type='SSE'
# method_type='no_scale'
# method_type='FGE'
# method_type='no scale'
method_type='Ind'

data_type='CIFAR10'
# data_type='CIFAR100'

# model_type='VGG19BN'
model_type='PreResNet110'

data = np.load(data_path)
data=data['ens_value.npy']


for  i  in range(data.shape[0]):
    data[i][i]=1

data_np=np.round(data,3)
# startcolor = '#000000'
# endcolor='#0000ff'
# cmap = col.LinearSegmentedColormap.from_list('own',[startcolor,endcolor])
# # extra arguments are N=256, gamma=1.0
# plt.cm.register_cmap(cmap=cmap)


fig, ax = plt.subplots()
im = ax.imshow(data_np,cmap=plt.cm.winter)
plt.colorbar(im)
width=range(data_np.shape[0])
# We want to show all ticks...
ax.set_xticks(width)
ax.set_yticks(width)
# ... and label them with the respective list entries
ax.set_xticklabels(width)
ax.set_yticklabels(width)


# Loop over data dimensions and create text annotations.
for i in width:
    for j in width:
        text = ax.text(j, i, data_np[i, j],
                       ha="center", va="center", color="black",fontsize=10 )

ax.set_title(method_type+'+'+model_type+'+'+data_type)
fig.tight_layout()
plt.savefig(data_path+method_type+model_type+data_type+'.pdf',dpi=600)
plt.show()