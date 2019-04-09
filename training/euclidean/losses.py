import torch as th
from torch import nn


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x,y):
        M, N = x.shape
        ix = th.arange(M, device=x.device)

        x = x.unsqueeze(1)
        eye = th.eye(N).to(x.device).unsqueeze_(0)

        diff = x-eye
        dist = diff.norm(2,dim=-1).pow(2)
        dist[ix,y]*=-1

        #l = th.exp(1/dist).sum(dim=-1).log().mean()
        l = th.exp(-dist).sum(dim=-1).log().mean()

        return l

class LogSumExpNegDist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        M, N = x.shape
        x = x.unsqueeze(1)
        eye = th.eye(N).to(x.device).unsqueeze_(0)

        diff = x-eye
        dist = diff.norm(2,dim=-1)

        l = th.exp(-dist).sum(dim=-1).log().mean()

        return l
