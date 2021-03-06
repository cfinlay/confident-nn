import pandas as pd
import os
import numpy as np
import argparse
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize


parser = argparse.ArgumentParser('Plot Odds (Bayes) Ratio for bins')

parser.add_argument('file', type=str,
        metavar='DF', 
        help='Location where pkl file saved')
parser.add_argument('--fig-size', type=float, default=4,
        help='Figure size (inches)')
parser.add_argument('--font-size',type=float, default=20)
parser.add_argument('--no-show', action='store_false', dest='show')
parser.add_argument('--show', action='store_true', dest='show')
parser.add_argument('--dpi', type=int, default=80)
parser.add_argument('--save', action='store_true', dest='save')
parser.add_argument('--no-save', action='store_false',dest='save')
parser.add_argument('--name', type=str, default='br.pdf', help='file name for saving')
parser.add_argument('--nbins', type=int, default=100)
parser.add_argument('--yvar', type=str, nargs='+', default=['model_entropy'])
parser.add_argument('--xvar', type=str, default='rank')
parser.add_argument('--xbins', type=float, default=[], nargs='*')
parser.add_argument('--ybins', type=float, default=[], nargs='*')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--eps', type=float, default=0)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--exclude', type=int, default=[], nargs='*')
parser.set_defaults(save=False)
parser.set_defaults(show=True)



from common import labdict

parser.set_defaults(show=True)
parser.set_defaults(save=False)

args = parser.parse_args()
np.random.seed(args.seed)

sns.set_palette(palette='colorblind')
colors = sns.color_palette()
cmap = ListedColormap(colors)
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

plt.close('all')
fig, ax = plt.subplots(1, figsize=figsz)

from common import labdict

print('X: %s, Y: %s'%(args.xvar, args.yvar))

df = pd.read_pickle(args.file)
df.drop(args.exclude)
Nsamples = len(df)


K = args.K
N = len(df)
Ix = np.random.permutation(N)

X_ = df[args.xvar]
for yvar in args.yvar:
    Y_ = df[yvar]

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




    Ptop5xbins = P[Xbins[:-1]<5,:].sum(axis=0)/Py
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

    BR1 = Otop1xbins/Otop1
    BR5 = Otop5xbins/Otop5

    BR1 = np.max([BR1,1/BR1],axis=0)
    BR5 = np.max([BR5,1/BR5],axis=0)


    ax.plot(BR5, 'o', label=labdict[yvar],ms=2)

ax.grid()
ax.set_axisbelow(True)
ax.set_yscale('log',nonposy='clip')
extra = []

extra.append(ax.set_xlabel('Bin'))
extra.append(ax.set_ylabel('Bayes ratio'))
fig.tight_layout()
labels = [labdict[lab] for lab in args.yvar]
leg_el = []
for c,lab in zip(colors[:len(labels)],labels):
    leg_el.append(Line2D([0],[0],markerfacecolor=c,markersize=10,marker='o', label=lab, color='w'))

#ax.legend(loc='upper right')
le =ax.legend(handles=leg_el,loc='left', bbox_to_anchor=(1,0.75))
extra.append(le)
#le =ax.legend(labels)#, loc='left', bbox_to_anchor=(1,0.5))



if show:
    plt.show()
if args.save:
    pth = os.path.split(args.file)[0]
    fig.savefig(os.path.join(pth,args.name),
            format='pdf',bbox_inches='tight',dpi=dpi)
