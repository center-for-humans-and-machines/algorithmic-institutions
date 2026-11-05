"""Surrogate v2: MLP over hand-made history features -> 21-class next contribution.
Trained on v1's sweep arrays (/tmp/pf7_arr.npz) + zero reference. Timeouts ignored."""
import numpy as np, torch as th, torch.nn as nn

def feats(C, P, t):
    """C [B,4,t+1], P [B,4,t+1] (punishment incl. round t) -> features [B,4,F] for predicting c(t+1)."""
    c0 = C[..., t]
    c1 = C[..., t-1] if t >= 1 else c0
    c2 = C[..., t-2] if t >= 2 else c1
    p0 = P[..., t]
    p1 = P[..., t-1] if t >= 1 else th.zeros_like(p0)
    p2 = P[..., t-2] if t >= 2 else th.zeros_like(p0)
    cum = P[..., :t+1].sum(-1); npun = (P[..., :t+1] > 0).float().sum(-1)
    mc = C[..., :t+1].mean(-1)
    def oth(x):
        return (x.sum(1, keepdim=True) - x) / 3
    big = 1e3
    def omax(x):
        m = x.unsqueeze(1).expand(-1, 4, -1).clone()
        idx = th.arange(4)
        m[:, idx, idx] = -big
        return m.max(-1).values
    def omin(x):
        m = x.unsqueeze(1).expand(-1, 4, -1).clone()
        idx = th.arange(4)
        m[:, idx, idx] = big
        return m.min(-1).values
    tt = th.full_like(c0, t / 23.)
    f = [c0/20, c1/20, c2/20, p0/30, p1/30, p2/30, cum/100, npun/10, mc/20,
         oth(c0)/20, oth(c1)/20, omax(c0)/20, omin(c0)/20, oth(p0)/30, omax(p0)/30, oth(cum)/100,
         (c0 == 20).float(), (c0 == 0).float(), tt, (p0 > 0).float()]
    return th.stack(f, -1)

class Net(nn.Module):
    def __init__(s, F=20, H=192):
        super().__init__()
        s.m = nn.Sequential(nn.Linear(F, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 21))
    def forward(s, x): return s.m(x)

def simulate(net, rule, C0, T=24, seed=0):
    """rule(C, P_hist, t) -> punishment [B,4]. C0: [B,4] round-0 contributions."""
    g = th.Generator().manual_seed(seed)
    B = C0.shape[0]
    C = C0.float().unsqueeze(-1); P = th.zeros(B, 4, 0)
    pay = 0.
    with th.no_grad():
        for t in range(T):
            p = rule(C, P, t).clamp(0, 30).floor()
            P = th.cat([P, p.unsqueeze(-1)], -1)
            c = C[..., t]
            payoff = 20 - c - p + (1.6 * c.sum(1, keepdim=True) - p.sum(1, keepdim=True)) / 4
            pay = pay + payoff.mean(1)
            if t < T - 1:
                x = feats(C, P, t)
                pr = th.softmax(net(x), -1)
                cn = th.multinomial(pr.reshape(-1, 21), 1, generator=g).reshape(B, 4).float()
                C = th.cat([C, cn.unsqueeze(-1)], -1)
    return pay, C, P
