import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import argparse

from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize

parser = argparse.ArgumentParser('Generate a scatter plot of two variables')

parser.add_argument('file', type=str,
        metavar='DF', 
        help='Location where pkl file saved')
parser.add_argument('--fig-size', type=float, default=6,
        help='Figure size (inches)')
parser.add_argument('--font-size',type=float, default=20)
parser.add_argument('--dpi', type=int, default=80)
parser.add_argument('--xvar', type=str, default='model_entropy')
parser.add_argument('--yvar', type=str, default='loss')
parser.add_argument('--no-show', action='store_false', dest='show')
parser.add_argument('--show', action='store_true', dest='show')
parser.add_argument('--save', action='store_true', dest='save')
parser.add_argument('--no-save', action='store_false',dest='save')
parser.add_argument('--leg', action='store_true', dest='leg')
parser.add_argument('--no-leg', action='store_false',dest='leg')
parser.add_argument('--log', action='store_true', dest='log')
parser.add_argument('--no-log', action='store_false',dest='log')
parser.add_argument('--name', type=str, default='scatter.pdf', help='file name for saving')
parser.add_argument('--xlim', type=float, default=None, nargs='*')
parser.add_argument('--ylim', type=float, default=None, nargs='*')
parser.set_defaults(show=True)
parser.set_defaults(log=True)
parser.set_defaults(save=False)
parser.set_defaults(leg=False)

args = parser.parse_args()

from common import labdict

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

df = pd.read_pickle(args.file)
Nsamples = len(df)

plt.close('all')

fig, ax = plt.subplots(1, figsize=figsz)
norm = Normalize(vmin=0,vmax=1.)
C = df['type'].map({'top1':0.2, 'top5':0.0, 'mis-classified':0.39})

X = df[args.xvar]
Y = df[args.yvar]

if args.xlim is None:
    xmin = max(0.9*X.min(), 1e-6)
    xmax = 1.5*X.max()
    scxlim = (xmin, xmax)
else:
    scxlim = tuple(args.xlim)

if args.ylim is None:
    ymin = max(0.9*Y.min(), 1e-6)
    ymax = 1.5*Y.max()
    scylim = (ymin, ymax)
else:
    scylim = tuple(args.ylim)


sc = ax.scatter(X,Y, c=C, s=1, cmap=cmap, norm=norm)

# ----------
# Set up GUI
# ----------
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join(list(map(str,ind["ind"]))))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(C[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

# ------------
# Minor tweaks
# ------------
ax.grid()
ax.set_axisbelow(True)
if args.log:
    ax.set_xscale('log',nonposx='clip')
ax.set_yscale('log',nonposy='clip')
ax.set_xlim(scxlim)
ax.set_ylim(scylim)

extra = []
extra.append(ax.set_xlabel(labdict[args.xvar]))
extra.append(ax.set_ylabel(labdict[args.yvar]))
fig.tight_layout()
if args.leg:
    leg_el = []
    labels = ['top5', 'mis-classified','top1']
    for c,lab in zip(colors[:3],labels):
        leg_el.append(Line2D([0],[0],markerfacecolor=c,markersize=10,marker='o', label=lab, color='w'))
    le =ax.legend(handles=leg_el,loc='left', bbox_to_anchor=(1,0.75))
    extra.append(le)


if show:
    plt.show()
if args.save:
    pth = os.path.split(args.file)[0]
    fig.savefig(os.path.join(pth,args.name),
            format='pdf',bbox_inches='tight',dpi=dpi)
