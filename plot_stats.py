# TODO: plot both top1 and top5 in different colors
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

sns.set_palette(palette='colorblind')
colors = sns.color_palette()
fsz = 20
figsz = (6,6)
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=.75*fsz)
plt.rc('figure', titlesize=fsz)

dpi = 80
show=True
fqylim=(0,0.2)
scxlim=(1.2e-6,80)
scylim=(0.9e-5,50)

modeldir = 'logs/imagenet/resnet152/'
data = os.path.join(modeldir,'eval.pkl')

df = pd.read_pickle(data)
Nsamples = len(df)

plt.close('all')
ix1 = np.array(df['top1'], dtype=bool)
ix5 = np.array(df['top5'], dtype=bool)
ix15 = np.logical_or(ix5,ix1)
ixw = np.logical_not(np.logical_or(ix1, ix5))

# ------------
# Scatter plot
# ------------
sj = 'gradx_modelsq_2norm'
Xw = df[ixw][sj]
Xc = df[ix1][sj]
Xc5 = df[ix5][sj]
st = 'loss'
Yc = df[ix1][st]
Yc5 = df[ix5][st]
Yw = df[ixw][st]

fig, ax = plt.subplots(1, figsize=figsz)

ax.scatter(Xw,Yw,s=0.2,c=colors[2])
ax.scatter(Xc5,Yc5,s=0.2,c=colors[0])
ax.scatter(Xc,Yc,s=0.2,c=colors[1])

ax.grid()
ax.set_axisbelow(True)
ax.set_xscale('log',nonposx='clip')
ax.set_yscale('log',nonposy='clip')
ax.set_xlim(scxlim)
ax.set_ylim(scylim)
ax.legend(['Mis-classified','top5 correct', 'top1 correct'],loc='best',markerscale=10)

extra = []
extra.append(ax.set_xlabel(r'$\Vert\nabla_x\vert f\vert^2 \Vert$'))
extra.append(ax.set_ylabel(r'$\ell(x)$'))
if show:
    plt.show()
fig.savefig(os.path.join(modeldir,'scatter.pdf'),
        format='pdf',bbox_inches='tight',dpi=dpi)




# ----------------------
# Frequency distribution
# ----------------------
X = df[sj]
fig, ax = plt.subplots(1, figsize=figsz)
ax.grid()
bins = np.logspace(-8,2,num=40)
ax.hist(X,bins=bins, weights=np.full(X.size,1/Nsamples), color=colors[2], label='Mis-classified')
ax.hist(X[ix15],bins=bins, weights=np.full(sum(ix15),1/Nsamples), color=colors[0], label='top5 correct')
ax.hist(X[ix1],bins=bins, weights=np.full(sum(ix1),1/Nsamples), color=colors[1], label='top1 correct')
ax.set_xscale('log',nonposx='clip')
extra = []
ax.set_ylim(fqylim)
ax.set_xlim(scxlim)
extra.append(ax.set_xlabel(r'$\Vert \nabla_x \vert f \vert^2 \Vert$'))
extra.append(ax.set_ylabel('frequency'))
ax.legend()
ax.set_axisbelow(True)
if show:
    plt.show()

fig.savefig(os.path.join(modeldir,'frequency.pdf'),
        format='pdf',bbox_inches='tight',dpi=dpi)
