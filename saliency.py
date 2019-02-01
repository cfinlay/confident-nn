"""Loads a pretrained model, then attacks it.
   Results are saved to a pkl file in the model's directory."""
import argparse
import os, sys

import pickle as pk
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import grad
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Subset

from flashlight import adversarial_training


parser = argparse.ArgumentParser('Compute loss gradients wrt ImageNet images.')

parser.add_argument('--data-dir', type=str,
        default='/mnt/data/scratch/ILSVRC2012/',metavar='DIR', 
        help='Directory where ImageNet data is saved')
parser.add_argument('--model', type=str, default='resnet152',
        choices=['resnet152'], help='Model')
parser.add_argument('--save-images', action='store_true', default=True,
        help='save perturbed images to a npy file')
parser.add_argument('--batch-size',type=int, default=100)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--indices', type=int, required=True, nargs='+')


#def main():
args = parser.parse_args()

if args.save_path is None:
    args.save_path = os.path.join('./logs/imagenet/',args.model)
pth = args.save_path
os.makedirs(pth, exist_ok=True)

args.batch_size = min(args.batch_size, len(args.indices))
args.num_images = len(args.indices)

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()

# Data loading code
valdir = os.path.join(args.data_dir, 'val')
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
clower = -np.array(mean)/np.array(std)
cupper = (1-np.array(mean))/np.array(std)

dataset = datasets.ImageFolder(valdir, transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize,
                                                    ]))
subset = Subset(dataset, args.indices)
loader = torch.utils.data.DataLoader( subset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
Nsamples = len(loader.dataset)
classes = 1000

model = getattr(models, args.model)(pretrained=True).cuda()
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


criterion = nn.CrossEntropyLoss()

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    criterion = criterion.cuda()


for i, (x,y) in enumerate(loader):
    if i==0:
        imshape = x.shape[1:]
        Gradients = torch.full((args.num_images, *imshape),np.nan)
        Images = torch.full((args.num_images, *imshape),np.nan)
        Labels = torch.full((args.num_images,),-1,dtype=torch.long)
        Predictions = torch.full((args.num_images,),-1,dtype=torch.long)
        if has_cuda:
            Gradients = Gradients.cuda()
            Images = Images.cuda()
            Labels = Labels.cuda()
            Predictions = Predictions.cuda()

    if has_cuda:
        x, y = x.cuda(), y.cuda()

    x = x.requires_grad_(True)
    yhat = model(x)
    pred = yhat.argmax(dim=-1)

    loss = criterion(yhat, y)
    dx = grad(loss, x)[0]

    Ix = torch.arange(i*args.batch_size, (i*args.batch_size+len(y)),device=x.device)
    Images[Ix] = x.detach()
    Gradients[Ix] = dx.detach()
    Labels[Ix] = y.detach()
    Predictions[Ix] = pred.detach()


if args.save_images:
    P = Images.cpu().numpy()
    O = Gradients.cpu().numpy()
    L = Labels.cpu().numpy()
    Pr = Predictions.cpu().numpy()

    with open(os.path.join(pth, 'saliency.pkl'),'wb') as f:
        pk.dump({'images':P, 'gradients':O, 'labels':L,'predictions':Pr,
            'indices':np.array(args.indices,dtype=np.int64)}, f)

#if __name__=="__main__":
#    main()

