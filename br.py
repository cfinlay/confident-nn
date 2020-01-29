import pandas as pd
import os
import numpy as np
import argparse
import warnings


parser = argparse.ArgumentParser('Bayes ratio and Brier score for histogram of two variables')

parser.add_argument('file', type=str,
        metavar='DF', 
        help='Location where pkl file saved')
parser.add_argument('--nbins', type=int, default=100)
parser.add_argument('--yvar', type=str, default='model_entropy')
parser.add_argument('--xvar', type=str, default='rank')
parser.add_argument('--xbins', type=float, default=[], nargs='*')
parser.add_argument('--ybins', type=float, default=[], nargs='*')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eps', type=float, default=0)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--exclude', type=int, default=[], nargs='*')




parser.set_defaults(show=True)
parser.set_defaults(save=False)

args = parser.parse_args()
np.random.seed(args.seed)


from common import labdict

print('X: %s, Y: %s'%(args.xvar, args.yvar))

df = pd.read_pickle(args.file)
df.drop(args.exclude)
Nsamples = len(df)


K = args.K
N = len(df)
Ix = np.random.permutation(N)

X_ = df[args.xvar]
Y_ = df[args.yvar]

EBR1 = []
EBR5 = []
#n = N//K
#ix = Ix[n*i:n*(i+1)]
#X = np.delete(X_.to_numpy(), ix)
#Y = np.delete(Y_.to_numpy(), ix)
X = X_[Ix]
Y = Y_[Ix]


Nbins = args.nbins
if len(args.ybins)==0:
    Yc, Ybins = pd.qcut(Y,Nbins,retbins=True,duplicates='drop')
else:
    Yc, Ybins = pd.cut(Y,args.ybins,retbins=True, duplicates='drop', right=False)

if len(args.xbins)==0:
    Xc, Xbins = pd.qcut(X,Nbins,retbins=True,duplicates='drop')
else:
    Xc, Xbins = pd.cut(X,args.xbins,retbins=True,duplicates='drop', right=False)

#Yvc = Yc.value_counts(sort=False)
#Xvc = Xc.value_counts(sort=False)


H, xe, ye = np.histogram2d(X, Y, bins=[Xbins, Ybins])

P = H/np.sum(H)




Ptop1 = df['top1'].sum()/len(df)
Ptop5 = df['top5'].sum()/len(df)
Otop1 = Ptop1/(1-Ptop1)
Otop5 = Ptop5/(1-Ptop5)

Py = P.sum(axis=0)
Ptop1xbins = P[Xbins[:-1]==0,:].reshape(-1)/Py
Brier1 = Ptop1xbins*(Ptop1xbins - 1)**2 + (1-Ptop1xbins)*Ptop1xbins**2

ix = np.arange(len(Ptop1xbins))
ix1 = Ptop1xbins==1
try:
    lb = np.max(ix[ix1])+1
except ValueError as e:
    lb = 0
Ptop1xbins[0:(lb+1)] = np.sum(Ptop1xbins[0:(lb+1)])/(lb+1)

ix0 = Ptop1xbins==0
try:
    ub = np.min(ix[ix0])
except ValueError as e:
    ub = len(Ptop1xbins)
Ptop1xbins[ub:] = np.sum(Ptop1xbins[ub:])/(len(Ptop1xbins)-ub+1)

Otop1xbins = Ptop1xbins/(1-Ptop1xbins+args.eps)
BR1 = Otop1xbins/Otop1

Ptop5xbins = P[Xbins[:-1]<5,:].sum(axis=0)/Py
Brier5 = Ptop5xbins*(Ptop5xbins - 1)**2 + (1-Ptop5xbins)*Ptop5xbins**2

ix5 = Ptop5xbins==1
try:
    lb = np.max(ix[ix5])+1
except ValueError as e:
    lb = 0
Ptop5xbins[0:(lb+1)] = np.sum(Ptop5xbins[0:(lb+1)])/(lb+1)

ix0 = Ptop5xbins==0
try:
    ub = np.min(ix[ix0])
except ValueError as e:
    ub = len(Ptop5xbins)
Ptop5xbins[ub:] = np.sum(Ptop5xbins[ub:])/(len(Ptop5xbins)-ub+1)

Otop5xbins = Ptop5xbins/(1-Ptop5xbins+args.eps)
BR5 = Otop5xbins/Otop5

BR1 = np.max([BR1,1/BR1],axis=0)
BR5 = np.max([BR5,1/BR5],axis=0)
EBR1.append(np.sum(Py*BR1))
EBR5.append(np.sum(Py*BR5))

print('E[Bayes ratio, top1] = %.3f'%np.mean(EBR1))
print('E[Bayes ratio, top5] = %.3f'%np.mean(EBR5))
print('\nBrier, top1 = %.3f'%np.sum(Py*Brier1))
print('Brier, top5 = %.3f'%np.sum(Py*Brier5))
