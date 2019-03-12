import pandas as pd
import os
import numpy as np
import argparse
import warnings


parser = argparse.ArgumentParser('Mutual information for histogram of two variables')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--nbins', type=int, default=100)
parser.add_argument('--yvar', type=str, default='model_entropy')
parser.add_argument('--xvar', type=str, default='rank')
parser.add_argument('--xbins', type=float, default=[], nargs='*')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--K', type=int, default=10)




parser.set_defaults(show=True)
parser.set_defaults(save=False)

args = parser.parse_args()
np.random.seed(args.seed)


from common import labdict

print('X: %s, Y: %s'%(args.xvar, args.yvar))

df = pd.read_pickle(args.file)
Nsamples = len(df)


K = args.K
N = len(df)
Ix = np.random.permutation(N)

X_ = df[args.xvar]
Y_ = df[args.yvar]

I = []
#for i in range(K):
    #n = N//K
    #ix = Ix[n*i:n*(i+1)]
    #X = np.delete(X_.to_numpy(), ix)
    #Y = np.delete(Y_.to_numpy(), ix)
X = X_[Ix]
Y = Y_[Ix]


Nbins = args.nbins
Yc, Ybins = pd.qcut(Y,Nbins,retbins=True, duplicates='drop')
if len(args.xbins)==0:
    Xc, Xbins = pd.qcut(X,Nbins,retbins=True,duplicates='drop')
else:
    Xc, Xbins = pd.cut(X,args.xbins,retbins=True,duplicates='drop', right=False)

#Yvc = Yc.value_counts(sort=False)
#Xvc = Xc.value_counts(sort=False)


H, xe, ye = np.histogram2d(X, Y, bins=[Xbins, Ybins])

P = H/np.sum(H)

Py = np.sum(P,axis=0,keepdims=True)
Px = np.sum(P,axis=1,keepdims=True)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    integrand = P*(np.log2(P) - np.log2(Px) - np.log2(Py))
integrand[np.isnan(integrand)]=0.
I.append(np.sum(integrand))

print('\nMutual information (bits): %.4g'%np.mean(I))
