import pandas as pd
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser('Generate a scatter plot')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--xvar', type=str, default='gradx_modelsq_2norm')
#parser.add_argument('--c', type=float, required=True)

args = parser.parse_args()


df = pd.read_pickle(args.file)
Nsamples = len(df)


wrong = df['type']=='mis-classified'
top5 = np.logical_not(wrong)
top1 = df['type']=='top1'
Nt5 = sum(top5)
Nt1 = sum(top1)

X = df[args.xvar]
I = np.argsort(X)
Xs = X[I]
t5s = top5[I]
t1s = top1[I]
ws = wrong[I]

N = np.arange(1,Nsamples+1)
Nt5 = t5s.cumsum()
Nw = ws.cumsum()
Nt1 = t1s.cumsum()

cdf = pd.DataFrame({'X': Xs, 'p(top5|x<X)': Nt5/N,
                    'p(wrong|x<X)':Nw/N,
                    'p(top1|x<X)':Nt1/N,
                    'p(x<X)': N/Nsamples})
