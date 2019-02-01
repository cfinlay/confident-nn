"""Loads a pretrained model, then attacks it.
   Results are saved to a pkl file in the model's directory."""
import argparse
import os, sys

import pickle as pk
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from flashlight import adversarial_training


parser = argparse.ArgumentParser('Test the stability of a model.'
                                  'Writes adversarial distances to a pkl file.')

parser.add_argument('--data-dir', type=str,
        default='/mnt/data/scratch/ILSVRC2012/',metavar='DIR', 
        help='Directory where ImageNet data is saved')
parser.add_argument('--model', type=str, default='resnet152',
        choices=['resnet152'], help='Model')
parser.add_argument('--num-images', type=int, default=1000,metavar='N',
        help='total number of images to attack (default: 1000)')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time')
parser.add_argument('--norm', type=str, default='L2',metavar='NORM', 
        choices=['L2','Linf','L1'],
        help='The norm measuring distances between images. (Gradient steps will be taken in the dual norm.) (default: "L2")')
parser.add_argument('--save-images', action='store_true', default=True,
        help='save perturbed images to a npy file')
parser.add_argument('--dt', type=float, default=0.01, help='step size (default: 0.01)')
parser.add_argument('--max-iters', type=int, default=200)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)


#def main():
args = parser.parse_args()
torch.manual_seed(args.seed)

if args.save_path is None:
    args.save_path = os.path.join('./logs/imagenet/',args.model)
pth = args.save_path
os.makedirs(pth, exist_ok=True)

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

loader = torch.utils.data.DataLoader(
                    datasets.ImageFolder(valdir, transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize,
                                                    ])),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
Nsamples = len(loader.dataset)
classes = len(loader.dataset.classes)

model = getattr(models, args.model)(pretrained=True).cuda()
model.eval()
for p in model.parameters():
    p.requires_grad_(False)


criterion = nn.CrossEntropyLoss()
d1 = torch.full((args.num_images,),np.inf)
d2 = torch.full((args.num_images,),np.inf)
dinf = torch.full((args.num_images,),np.inf)

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    criterion = criterion.cuda()
    d1 = d1.cuda()
    d2 = d2.cuda()
    dinf = dinf.cuda()

if args.norm=='L2':
    perturb = adversarial_training.L2Perturbation(model, args.dt, criterion)
if args.norm=='Linf':
    perturb = adversarial_training.L1Perturbation(model, args.dt, criterion)
if args.norm=='L1':
    perturb = adversarial_training.LinfPerturbation(model, args.dt, criterion)



for i, (x,y) in enumerate(loader):
    if i==0:
        imshape = x.shape[1:]
        if args.save_images:
            PerturbedImages = torch.full((args.num_images, *imshape),np.nan)
            OrigImages = torch.full((args.num_images, *imshape),np.nan)
            Labels = torch.full((args.num_images,),-1,dtype=torch.long)
            if has_cuda:
                PerturbedImages = PerturbedImages.cuda()
                OrigImages = OrigImages.cuda()
                Labels = Labels.cuda()
    if i*args.batch_size >= args.num_images:
        break
    if has_cuda:
        x,y  = x.cuda(), y.cuda()
    xorig = x.clone()
    Ix = torch.arange(i*args.batch_size, (i*args.batch_size+len(y)),device=x.device)
    pert = torch.zeros_like(x)
    switched = torch.zeros_like(y).byte()
    k = 0

    while k<args.max_iters and len(switched[switched])<len(y):
        x = x.detach()
        x.requires_grad_(True)

        yhat = model(x).topk(1)[1].view(-1).detach()
        c = (yhat==y).detach()
        ix = ~(switched | c).detach()
        pert[ix] = x[ix].detach()
        switched = (switched | ~c).detach()

        x = perturb(x,y,prep=False)
        x[:,0,:,:].clamp_(clower[0],cupper[0])
        x[:,1,:,:].clamp_(clower[1],cupper[1])
        x[:,2,:,:].clamp_(clower[2],cupper[2])

        k+=1

    diff = xorig.detach() - pert.detach()
    l1 = diff.view(len(y), -1).norm(p=1, dim=-1)
    l2 = diff.view(len(y), -1).norm(p=2, dim=-1)
    linf = diff.view(len(y), -1).norm(p=np.inf, dim=-1)

    sys.stdout.write('[Batch %2d/%3d] median & max distance: (%4.4f, %4.4f)\r'
            %(i+1, args.num_images//args.batch_size, l2.median(),l2.max()))
    sys.stdout.flush()

    d1[Ix[switched]] = l1[switched]
    d2[Ix[switched]] = l2[switched]
    dinf[Ix[switched]] = linf[switched]
    if args.save_images:
        PerturbedImages[Ix[switched]] = pert.detach()[switched]
        OrigImages[Ix] = xorig.detach()
        Labels[Ix] = y

sys.stdout.write('\n   Done\n')

dists = pd.DataFrame({'l1': d1.cpu().numpy(),
         'l2': d2.cpu().numpy(),
         'linf': dinf.cpu().numpy()})
dists.to_pickle(os.path.join(pth,args.norm+'-PGD-attack-distances.pkl'))

if args.save_images:
    P = PerturbedImages.cpu().numpy()
    O = OrigImages.cpu().numpy()
    L = Labels.cpu().numpy()

    with open(os.path.join(pth, args.norm+'-PGD-attack-images.pkl'),'wb') as f:
        pk.dump({'original':O, 'perturbed':P, 'labels':L}, f)

#if __name__=="__main__":
#    main()
