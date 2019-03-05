import pandas as pd
import os
import numpy as np
import argparse
import warnings


parser = argparse.ArgumentParser('Mutual information for histogram of two variables')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--nbins', type=int, default=50)
parser.add_argument('--xvar', type=str, default='loss')
parser.add_argument('--yvar', type=str, default='model_entropy')

parser.set_defaults(show=True)
parser.set_defaults(save=False)

args = parser.parse_args()

from common import labdict


df = pd.read_pickle(args.file)
Nsamples = len(df)


X = df[args.xvar]#[b]
Y = df[args.yvar]#[b]

Nbins = args.nbins
Yc, Ybins = pd.qcut(Y,Nbins,retbins=True, duplicates='drop')
Xc, Xbins = pd.qcut(X,Nbins,retbins=True,duplicates='drop')
Yvc = Yc.value_counts(sort=False)
Xvc = Xc.value_counts(sort=False)


H, xe, ye = np.histogram2d(X, Y, bins=[Xbins, Ybins])
Hy = np.sum(H,axis=0,keepdims=True)
Hx = np.sum(H,axis=1,keepdims=True)

dx = (Xbins[1:]-Xbins[:-1])
dx = np.reshape(dx, Hx.shape)
dy = (Ybins[1:]-Ybins[:-1])
dy = np.reshape(dy, Hy.shape)

dX = dx*dy
N = len(df)
P = H/N
Px = Hx/N
Py = Hy/N




with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    integrand = P*(np.log(P) - np.log(Px) - np.log(Py))
integrand[np.isnan(integrand)]=0.
I = np.sum(integrand*dX)

print('\nMutual information: %.4g\n'%I)
