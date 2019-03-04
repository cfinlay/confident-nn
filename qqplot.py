import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse


parser = argparse.ArgumentParser('QQ Plot for histograms of two variables')

parser.add_argument('--file', type=str,
        default='logs/imagenet/resnet152/eval.pkl',metavar='F', 
        help='Location where pkl file saved')
parser.add_argument('--fig-size', type=float, default=6,
        help='Figure size (inches)')
parser.add_argument('--font-size',type=float, default=20)
parser.add_argument('--dpi', type=int, default=80)
parser.add_argument('--nbins', type=int, default=10)
parser.add_argument('--xvar', type=str, default='loss')
parser.add_argument('--yvar', type=str, default='model_entropy')
parser.add_argument('--no-show', action='store_false', dest='show')
parser.add_argument('--show', action='store_true', dest='show')
parser.add_argument('--save', action='store_true', dest='save')
parser.add_argument('--no-save', action='store_false',dest='save')
parser.add_argument('--name', type=str, default='qqplot.pdf')
parser.add_argument('--xlim', type=float, default=None, nargs='*')
parser.add_argument('--ylim', type=float, default=None, nargs='*')
parser.add_argument('--filter', type=str, choices=['all','top1','top5','wrong'],default='all')

parser.set_defaults(show=True)
parser.set_defaults(save=False)

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
plt.rc('legend', fontsize=.75*fsz)
plt.rc('figure', titlesize=fsz)

dpi = args.dpi
show=args.show

df = pd.read_pickle(args.file)
Nsamples = len(df)

plt.close('all')

fig, ax = plt.subplots(1, figsize=figsz)

if args.filter in ['top1','top5']:
    b = df[args.filter]
elif args.filter=='wrong':
    b = np.logical_not(df['top5'])
else:
    b = np.arange(len(df))
X = df[args.xvar][b]
Y = df[args.yvar][b]
X = X[X>0]
Y = Y[Y>0]
X = np.log(X)
Y = np.log(Y)

Nbins = args.nbins
Yc, Ybins = pd.qcut(Y,Nbins,retbins=True)
Xc, Xbins = pd.qcut(X,Nbins,retbins=True)
Yvc = Yc.value_counts(sort=False)
Xvc = Xc.value_counts(sort=False)

fit, residuals, rank, sv, rcond = np.polyfit(Xbins[1:], Ybins[1:],1, full=True)
Ypred= np.poly1d(fit)(Xbins[1:])

m = np.mean(Ybins[1:])
SStot = np.sum((Ybins[1:] - m)**2)
Rsq = 1-residuals[0]/SStot
print('QQ-Plot R^2: %.4g'%Rsq)


#ax.plot(np.exp(Xbins[1:]), np.exp(Ypred), color='k')
ax.plot(np.exp(Xbins[1:]), np.exp(Xbins[1:]), color='k')
ax.scatter(np.exp(Xbins[1:]), np.exp(Ybins[1:]))
ax.grid()
ax.set_axisbelow(True)

if args.xlim is None:
    pass
    #ax.set_xlim((np.exp(0.5*Xbins[1]),np.exp(1.5*Xbins[-1])))
else:
    ax.set_xlim(*args.xlim)

if args.ylim is None:
    pass
    #ax.set_ylim(np.exp(0.5*Ybins[1]),np.exp(1.5*Ybins[-1]))
else:
    ax.set_ylim(*args.ylim)

ax.set_xscale('log',nonposx='clip')
ax.set_yscale('log',nonposy='clip')

extra = []
extra.append(ax.set_xlabel(labdict[args.xvar]+' quantiles'))
extra.append(ax.set_ylabel(labdict[args.yvar]+' quantiles'))
fig.tight_layout()
#
if show:
    plt.show()
#    
if args.save:
    pth = os.path.split(args.file)[0]
    fig.savefig(os.path.join(pth,args.name),
            format='pdf',bbox_inches='tight',dpi=dpi)

