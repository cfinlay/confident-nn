import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse


parser = argparse.ArgumentParser('Generate a frequency histogram')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--fig-size', type=float, default=6,
        help='Figure size (inches)')
parser.add_argument('--font-size',type=float, default=20)
parser.add_argument('--dpi', type=int, default=80)
parser.add_argument('--nbins', type=int, default=40)
parser.add_argument('--equal', action='store_true', default=False)
parser.add_argument('--xvar', type=str, default='model_entropy')
parser.add_argument('--ylabel', type=str, default=None)
parser.add_argument('--yscale', type=float, default=1.)
parser.add_argument('--no-show', action='store_false', dest='show')
parser.add_argument('--show', action='store_true', dest='show')
parser.add_argument('--save', action='store_true', dest='save')
parser.add_argument('--no-save', action='store_false',dest='save')
parser.add_argument('--leg', action='store_true', dest='leg')
parser.add_argument('--no-leg', action='store_false',dest='leg')
parser.add_argument('--name', type=str, default='frequency.pdf')
parser.add_argument('--xlim', type=float, default=None, nargs='*')
parser.add_argument('--ylim', type=float, nargs='*', default=(0,0.2))
parser.set_defaults(show=True)
parser.set_defaults(save=False)
parser.set_defaults(leg=False)

args = parser.parse_args()

from common import labdict

sns.set_palette(palette='colorblind')
colors = sns.color_palette()
fsz = args.font_size
figsz = (args.fig_size, args.fig_size)
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=.66*fsz)
plt.rc('figure', titlesize=fsz)

dpi = args.dpi
show=args.show

df = pd.read_pickle(args.file)
Nsamples = len(df)

plt.close('all')

fig, ax = plt.subplots(1, figsize=figsz)

X = df[args.xvar]

if args.xlim is None:
    xmin = max(0.9*X.min(), 1e-6)
    xmax = 1.5*X.max()
    scxlim = (xmin, xmax)
else:
    scxlim = tuple(args.xlim)
    xmin, xmax = args.xlim

scylim = tuple(args.ylim)

X = df[args.xvar]
ix1 = np.array(df['top1'], dtype=bool)
ix5 = np.array(df['top5'], dtype=bool)
ix15 = np.logical_or(ix5,ix1)
ixw = np.logical_not(np.logical_or(ix1, ix5))


if not args.equal:
    bins = np.logspace(np.log10(xmin),np.log10(xmax),num=args.nbins)
else:
    Xc, bins = pd.qcut(X,args.nbins,retbins=True,duplicates='drop')

ax.hist(X,bins=bins, weights=args.yscale*np.full(X.size,1/Nsamples), color=colors[1], label='mis-classified')
ax.hist(X[ix15],bins=bins, weights=args.yscale*np.full(sum(ix15),1/Nsamples), color=colors[0], label='top5')
ax.hist(X[ix1],bins=bins, weights=args.yscale*np.full(sum(ix1),1/Nsamples), color=colors[2], label='top1')

ax.grid()
ax.set_axisbelow(True)
ax.set_xscale('log',nonposx='clip')
ax.set_xlim(scxlim)
ax.set_ylim(scylim)

if args.leg:
    ax.legend(loc='best')

extra = []
extra.append(ax.set_xlabel(labdict[args.xvar]))
if args.ylabel is None:
    extra.append(ax.set_ylabel('frequency'))
else:
    extra.append(ax.set_ylabel(args.ylabel))

fig.tight_layout()

if show:
    plt.show()
    
if args.save:
    pth = os.path.split(args.file)[0]
    fig.savefig(os.path.join(pth,args.name),
            format='pdf',bbox_inches='tight',dpi=dpi)
