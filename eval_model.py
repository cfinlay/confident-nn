import argparse
import os, sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import grad




parser = argparse.ArgumentParser('Gathers stability/robustness stats of an ImageNet model.')

parser.add_argument('--model', type=str, default='resnet152',
        choices=['resnet152'], help='Model')
parser.add_argument('--data-dir', type=str,
        default='/mnt/data/scratch/ILSVRC2012/',metavar='DIR', 
        help='Directory where ImageNet data is saved')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to evaluate at a time')
parser.add_argument('--mode', type=str,default='val', choices=['val'])
parser.add_argument('--save-path', type=str, default=None)

def main():
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = os.path.join('./logs/imagenet/',args.model, 'eval.pkl')
    pth = os.path.split(args.save_path)[0]
    os.makedirs(pth, exist_ok=True)

    print('Arguments:')
    for p in vars(args).items():
        print('  ',p[0]+': ',p[1])
    print('\n')


    # Data loading code
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #train_loader = torch.utils.data.DataLoader(
    #                    datasets.ImageFolder(traindir, transforms.Compose([
    #                                            transforms.RandomResizedCrop(224),
    #                                            transforms.RandomHorizontalFlip(),
    #                                            transforms.ToTensor(),
    #                                            normalize,
    #                                                ])),
    #                    batch_size=args.batch_size, shuffle=True,
    #                    num_workers=4, pin_memory=True)

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

    m = getattr(models, args.model)(pretrained=True).cuda()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
        
    if torch.cuda.device_count()>1:
        m = nn.DataParallel(m)

    class GradNorms(nn.Module):
        def __init__(self, norm, summary=lambda x: x, create_graph=False, retain_graph=False):
            super().__init__()
            self.norm = norm
            self.create_graph=create_graph
            self.summary = summary
            self.retain_graph = retain_graph

        def forward(self, l, x):
            sh = x.shape
            bsz = sh[0]

            if not x.requires_grad:
                x.requires_grad_()

            dx, = grad(l, x, create_graph=self.create_graph, retain_graph=self.retain_graph)
            dx = dx.view(bsz, -1)
            
            if self.norm in [2,'2']:
                n = dx.norm(p=2,dim=-1)
            elif self.norm in [1,'1']:
                n = dx.norm(p=1,dim=-1)
            elif self.norm in ['inf',inf]:
                n = dx.abs().max(dim=-1)[0]
            else:
                raise ValueError('%s is not an available norm'%self.norm)

            return self.summary(n)

    GradFunc0 = GradNorms(2,retain_graph=True)
    GradFunc1 = GradNorms(2,retain_graph=False)

    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    Loss = torch.zeros(Nsamples).cuda()
    Top1 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()
    Top5 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()
    LossGradx = torch.zeros(Nsamples).cuda()
    ModelSqGradx = torch.zeros(Nsamples).cuda()
    ModelSqGradw = torch.zeros(Nsamples).cuda()
    LossGradw = torch.zeros(Nsamples).cuda()
    LossGrady = torch.zeros(Nsamples).cuda()



    sys.stdout.write('\nRunning through dataloader:\n')
    for i, (x,y) in enumerate(loader):
        sys.stdout.write('  Completed [%6.2f%%]\r'%(100*i*args.batch_size/Nsamples))
        sys.stdout.flush()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        yhat = m(x)
        pnorm = yhat.softmax(dim=-1).norm(dim=-1)
        loss = criterion(yhat, y)

        gx = GradFunc0(loss.sum(), x)
        dpn = GradFunc1(pnorm.sum(),x)
        gy = (-yhat.softmax(dim=-1).log()).norm(dim=-1)


        top1 = torch.argmax(yhat,dim=-1)==y
        s = yhat.sort(dim=-1, descending=True)[1]
        top5 = (s[:,0:5]==y.view(args.batch_size,1)).sum(dim=-1)

        ix = torch.arange(i*args.batch_size, (i+1)*args.batch_size,device=x.device)

        Loss[ix] = loss.detach()
        Top1[ix] = top1.detach()
        Top5[ix] = top5.detach().type(torch.uint8)
        LossGradx[ix] = gx.detach()
        ModelSqGradx[ix] = dpn.detach()
        LossGrady[ix] = gy.detach()
    sys.stdout.write('   Completed [%6.2f%%]\r'%(100.))

    loader1 = torch.utils.data.DataLoader(
                        datasets.ImageFolder(valdir, transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                                        ])),
                        batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    sys.stdout.write('\n\nChecking loss gradients wrt weights:\n')
    if torch.cuda.device_count()>1:
        m = m.module

    for p in m.parameters():
        p.requires_grad_(True)
    for i, (x,y) in enumerate(loader1):
        sys.stdout.write('  Completed [%6.2f%%]\r'%(100*i/Nsamples))
        sys.stdout.flush()

        x, y = x.cuda(), y.cuda()


        yhat = m(x)
        loss = criterion(yhat, y)
        loss.backward()
        gwsq = 0.
        for p in m.parameters():
            gwsq += p.grad.view(-1).pow(2).sum()
            p.grad.detach_()
            p.grad.zero_()
        gw = gwsq.sqrt()
        LossGradw[i] = gw

        yhat = m(x)
        pnorm = yhat.softmax(dim=-1).norm(dim=-1)
        pnorm.backward()
        gwsq = 0.
        for p in m.parameters():
            gwsq += p.grad.view(-1).pow(2).sum()
            p.grad.detach_()
            p.grad.zero_()
        gw = gwsq.sqrt()
        ModelSqGradw[i] = gw

    sys.stdout.write('   Completed [%6.2f%%]\r'%(100.))
    sys.stdout.write('\n   Done')
    sys.stdout.flush()

    df = pd.DataFrame({'loss':Loss.cpu().numpy(),
                       'top1':np.array(Top1.cpu().numpy(),dtype=np.bool),
                       'top5':np.array(Top5.cpu().numpy(), dtype=np.bool),
                       'gradx_loss_2norm': LossGradx.cpu().numpy(),
                       'gradx_modelsq_2norm': ModelSqGradx.cpu().numpy(),
                       'gradw_modelsq_2norm': ModelSqGradw.cpu().numpy(),
                       'gradw_loss_2norm': LossGradw.cpu().numpy(),
                       'grady_loss_2norm': LossGrady.cpu().numpy()})


    df.to_pickle(args.save_path)

if __name__=='__main__':
    main()