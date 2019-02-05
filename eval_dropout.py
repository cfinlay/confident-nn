import argparse
import os, sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import models.resnet as models
from torch.autograd import grad




parser = argparse.ArgumentParser('Reports variance of ImageNet model ran with dropout.')

parser.add_argument('--model', type=str, default='resnet152',
        choices=['resnet152'], help='Model')
parser.add_argument('--data-dir', type=str,
        default='/mnt/data/scratch/ILSVRC2012/',metavar='DIR', 
        help='Directory where ImageNet data is saved')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to evaluate at a time')
parser.add_argument('--mode', type=str,default='val', choices=['val'])
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--p', type=float, defaut=0.01)
parser.add_argument('--reps', type=int, defaut=30)

def main():
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = os.path.join('./logs/imagenet/',args.model, 'eval.pkl')
    pth = os.path.split(args.save_path)[0]
    os.makedirs(pth, exist_ok=True)


    # Data loading code
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    val_loader = torch.utils.data.DataLoader(
                        datasets.ImageFolder(valdir, transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                                        ])),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    loader = val_loader# if args.mode=='val' else train_loader
    Nsamples = len(loader.dataset)
    Nclasses = len(loader.dataset.classes)


    Nreps = args.reps
    dropout = args.p
    m = getattr(models, args.model)(pretrained=True, dropout=dropout).cuda()
    m.eval()

    def turn_on_dropout(module):
        if type(module)==nn.Dropout:
            module.train()

    m.apply(turn_on_dropout)


    for p in m.parameters():
        p.requires_grad_(False)
    if torch.cuda.device_count()>1:
        m = nn.DataParallel(m)



    sys.stdout.write('\nRunning through dataloader:\n')
    VarNorm = torch.zeros(Nsamples).cuda()
    k = 0
    for i, (x,y) in enumerate(loader):

        x, y = x.cuda(), y.cuda()
        Nb = len(y)
        ix = torch.arange(k, k+Nb).cuda()

        mean = torch.zeros(Nb, Nclasses).cuda()
        moment2 = torch.zeros(Nb, Nclasses, Nclasses).cuda()


        for j in range(Nreps):
            sys.stdout.write('   [Batch %2d, rep %2d]\r'%(i,j))
            sys.stdout.flush()
            yhat = m(x)
            p = yhat.softmax(dim=-1)

            mean = (mean*j+ p)/(j+1) 
            outer = torch.einsum('ij,ik->ijk',(p,p))
            moment2 = (moment2*j + outer)/(j+1)


        mouter = torch.einsum('ij,ik->ijk',(mean,mean))
        var = moment2 - mouter

        varview = var.view(Nb, -1)
        frob = varview.norm(2,-1)
        VarNorm[ix] = frob



    sys.stdout.write('\n   Done\n')

    try:
        df = pd.read_pickle(os.path.join(args.save_path))
        df['norm_dropout_var'] = VarNorm.cpu().numpy()
    except:
        df = pd.DataFrame({'norm_dropout_var':VarNorm.cpu().numpy()})

    df.to_pickle(args.save_path)

if __name__=='__main__':
    main()
