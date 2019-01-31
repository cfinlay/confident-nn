import pandas as pd
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser('Generate a scatter plot')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--fig-size', type=float, default=6,
        help='Figure size (inches)')
parser.add_argument('--xvar', type=str, default='gradx_modelsq_2norm')
parser.add_argument('--p', type=float, default=0.05)

args = parser.parse_args()
p = args.p



df = pd.read_pickle(args.file)
Nsamples = len(df)


wrong = df['type']=='mis-classified'
top5 = np.logical_not(wrong)
top1 = df['type']=='top1'
Nt5 = sum(top5)
Nt1 = sum(top1)

X = df[args.xvar]
thresh = X[wrong].quantile(p)

false5 = (X[top5]>=thresh).sum()/Nt5
false1 = (X[top1]>=thresh).sum()/Nt1

print('Hypothesis: Image misclassified')
print('    Variable: %s, alpha=%.2f%%'%(args.xvar, args.p*100))
print('')
print('    P(image top5 but rejected) = %.2f%%'%(false5*100))
print('    P(image top1 but rejected) = %.2f%%'%(false1*100))








