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
parser.add_argument('--quintiles', type=float, nargs='+', default=[0.99,0.9])
parser.add_argument('--yvar', type=str, default='model_entropy')
parser.add_argument('--xvar', type=str, default='rank')



parser.set_defaults(show=True)
parser.set_defaults(save=False)

args = parser.parse_args()

q0, q1 = args.quintiles

from common import labdict

print('X: %s, Y: %s'%(args.xvar, args.yvar))

df = pd.read_pickle(args.file)
Nsamples = len(df)


# TODO: k-fold cross-validation

X = df[args.xvar]
Y = df[args.yvar]

Nbins = args.nbins
Yc, Ybins = pd.qcut(Y,Nbins,retbins=True, duplicates='drop')
Xc, Xbins = pd.qcut(X,Nbins,retbins=True,duplicates='drop')
Yvc = Yc.value_counts(sort=False)
Xvc = Xc.value_counts(sort=False)


H, xe, ye = np.histogram2d(X, Y, bins=[Xbins, Ybins])

P = H/np.sum(H)

Py = np.sum(P,axis=0,keepdims=True)
Px = np.sum(P,axis=1,keepdims=True)


with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    integrand = P*(np.log2(P) - np.log2(Px) - np.log2(Py))
integrand[np.isnan(integrand)]=0.
I = np.sum(integrand)

print('\nMutual information (bits): %.4g\n'%I)


if args.xvar=='rank':
    # ---------
    # Top1 bins
    # ---------
    CDFx = np.cumsum(P, axis=1)
    TotMass = np.cumsum(Py, axis=1)
    Conditional = CDFx/TotMass
    ConditionalTop1 = Conditional[0,:]
    cutix=np.nonzero(ConditionalTop1>q0)[0][-1]


    Nc = cutix+1
    CDFx2 = np.cumsum(P[:,Nc:], axis=1)
    TotMass2 = np.cumsum(Py[:,Nc:], axis=1)
    Conditional2 = CDFx2/TotMass2
    ConditionalTop12 = Conditional2[0,:]
    cutix2=np.nonzero(ConditionalTop12>q1)[0][-1]


    Nc2 = cutix+cutix2+2
    CDFx3 = np.cumsum(P[:,Nc2:], axis=1)
    TotMass3 = np.cumsum(Py[:,Nc2:], axis=1)
    Conditional3 = CDFx3/TotMass3
    ConditionalTop13 = Conditional3[0,:]


    Bins = np.array([Ybins[Nc],Ybins[Nc2]])

    print('P(top1 | Y < %.2g) = %.2g,\t\t P(Y < %.2g) = %.2g'%(Bins[0],q0,Bins[0],TotMass[:,cutix][0]))
    print('P(top1 | %.2g <= Y < %.2g) = %.2g,\t P(%.2g <= Y < %.2g) = %.2g'%(Bins[0],Bins[1],q1,Bins[0],Bins[1], TotMass2[:,cutix2][0]))
    print('P(top1 | Y >= %.2g) = %.2g,\t\t P(Y >= %.2g) = %.2g'%(Bins[1],ConditionalTop13[-1],Bins[1], TotMass3[:,-1][0]))

    # ---------
    # Top5 bins
    # ---------
    CDFx = np.cumsum(P, axis=1)
    TotMass = np.cumsum(Py, axis=1)
    Conditional = CDFx/TotMass
    ConditionalTop5 = np.sum(Conditional[0:5,:],axis=0)
    cutix5=np.nonzero(ConditionalTop5>q0)[0][-1]


    Nc5 = cutix5+1
    CDFx52 = np.cumsum(P[:,Nc5:], axis=1)
    TotMass52 = np.cumsum(Py[:,Nc5:], axis=1)
    Conditional52 = CDFx52/TotMass52
    ConditionalTop52 = np.sum(Conditional52[0:5,:],axis=0)
    cutix52=np.nonzero(ConditionalTop52>q1)[0][-1]


    Nc52 = cutix5+cutix52+2
    CDFx53 = np.cumsum(P[:,Nc52:], axis=1)
    TotMass53 = np.cumsum(Py[:,Nc52:], axis=1)
    Conditional53 = CDFx53/TotMass53
    ConditionalTop53 = Conditional53[0,:]


    Bins5 = np.array([Ybins[Nc5],Ybins[Nc52]])

    print('\nP(top5 | Y < %.2g) = %.2g,\t\t P(Y < %.2g) = %.2g'%(Bins5[0],q0,Bins5[0],TotMass[:,cutix5][0]))
    print('P(top5 | %.2g <= Y < %.2g) = %.2g,\t P(%.2g <= Y < %.2g) = %.2g'
            %(Bins5[0],Bins5[1],q1,Bins5[0],Bins5[1], TotMass52[:,cutix52][0] ))
    print('P(top5 | Y >= %.2g) = %.2g,\t\t P(Y >= %.2g) = %.2g'%(Bins5[1],ConditionalTop53[-1], Bins5[1], TotMass53[:,-1][0]))

