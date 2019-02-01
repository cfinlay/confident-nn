import pickle as pk
import numpy as np
import json
import pandas as pd

import torch as th
import torchvision.utils as utils
import matplotlib.pyplot as plt

pth = 'logs/imagenet/resnet152/saliency.pkl'
with open(pth,'rb') as f:
    dct = pk.load(f)

class_idx = json.load(open("imagenet_class_index.json",'r'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

predictions = [idx2label[i] for i in dct['predictions']]
labels = [idx2label[i] for i in dct['labels']]

df = pd.DataFrame({'labels':labels, 'prediction':predictions})
print(df)


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

X = dct['images']
X = th.from_numpy(X)
X[:,0,:,:].mul_(std[0]).add_(mean[0])
X[:,1,:,:].mul_(std[1]).add_(mean[1])
X[:,2,:,:].mul_(std[2]).add_(mean[2])

#dX = dct['gradients']
#dX = th.from_numpy(dX)

#N = len(dX)
#sh = dX.shape
#dXv = dX.view(N,-1)
#a,b = dXv.min(dim=-1, keepdim=True)[0], dXv.max(dim=-1,keepdim=True)[0]
#dXv.add_(-a).div_(b-a)
#dX = dXv.view(*sh)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    
#Ims = th.cat([X,dX])
Ims = X
g = utils.make_grid(Ims,nrow=5)
show(g)
plt.show()
