import pandas as pd
import os
import numpy as np
import argparse
import warnings


parser = argparse.ArgumentParser('Mutual information for histogram of two variables')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--nbins', type=int, default=40)
parser.add_argument('--xvar', type=str, default='loss')
parser.add_argument('--yvar', type=str, default='model_entropy')
parser.add_argument('--filter', type=str, choices=['all','top1','top5','wrong'],default='all')

parser.set_defaults(show=True)
parser.set_defaults(save=False)

args = parser.parse_args()

from common import labdict


df = pd.read_pickle(args.file)
Nsamples = len(df)

if args.filter in ['top1','top5']:
    b = df[args.filter]
elif args.filter=='wrong':
    b = np.logical_not(df['top5'])
else:
    b = np.arange(len(df))
X = df[args.xvar][b]
Y = df[args.yvar][b]
X_ = X[np.logical_and(X>0, Y>0)]
Y = Y[np.logical_and(X>0,Y>0)]
X = X_

df = pd.DataFrame({'logX':np.log(X),'logY':np.log(Y)})

H, xe, ye = np.histogram2d(df['logX'], df['logY'], bins=args.nbins, density=True)
H = H/np.sum(H)
Hx = np.sum(H,axis=0,keepdims=True)
Hy = np.sum(H,axis=1,keepdims=True)

dx = (xe[1:]-xe[:-1])[0]
dy = (ye[1:]-ye[:-1])[0]



with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    integrand = H*(np.log(H) - np.log(Hx) - np.log(Hy))
integrand[np.isnan(integrand)]=0.
I = np.sum(integrand)*dx*dy

print('\nMutual information: %.4g\n'%I)
