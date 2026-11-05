"""Surrogate v3: MLP over richer history features (vectorised over all rounds) -> 21-class next c.
Trained on real sweep games of v1 (+ v2). Timeouts ignored (c of invalid rows kept as recorded)."""
import torch as th, torch.nn as nn

NF = 30
def feats_all(C, P):
    """C,P [B,4,T] float -> features [B,4,T,NF], feature at t for predicting c(t+1)."""
    B, N, T = C.shape
    def lag(x, k, fill=None):
        if k == 0: return x
        f = x[..., :1].expand(-1, -1, k) if fill is None else th.full_like(x[..., :1].expand(-1, -1, k), fill)
        return th.cat([f, x], -1)[..., :x.shape[-1]]
    ar = th.arange(1, T + 1, dtype=C.dtype)
    cumc = C.cumsum(-1); mc = cumc / ar
    c5 = (cumc - lag(cumc, 5, 0.)) / th.clamp(ar, max=5)
    cmax = th.cummax(C, -1).values
    cum = P.cumsum(-1); npun = (P > 0).float().cumsum(-1); pmax = th.cummax(P, -1).values
    # rounds since last fine
    idx = th.where(P > 0, ar.expand_as(P), th.zeros_like(P))
    last = th.cummax(idx, -1).values
    since = th.where(last > 0, ar - last, th.full_like(P, 25.))
    oth = lambda x: (x.sum(1, keepdim=True) - x) / 3
    # sorted others' c
    Ce = C.unsqueeze(1).expand(B, N, N, T).clone()
    eye = th.eye(N, dtype=th.bool).view(1, N, N, 1).expand(B, N, N, T)
    Ce[eye] = -1.
    srt = Ce.sort(2).values[:, :, 1:, :]  # 3 others ascending
    Pe = P.unsqueeze(1).expand(B, N, N, T).clone(); Pe[eye] = -1.
    opmax = Pe.max(2).values
    tt = (ar - 1).expand_as(C) / 23.
    f = [C/20, lag(C,1)/20, lag(C,2)/20, c5/20, mc/20, cmax/20,
         P/30, lag(P,1,0.)/30, lag(P,2,0.)/30, cum/100, npun/10, pmax/30, since/25,
         oth(C)/20, oth(lag(C,1))/20, srt[:,:,0]/20, srt[:,:,1]/20, srt[:,:,2]/20,
         oth(P)/30, opmax/30, oth(cum)/100, oth(lag(P,1,0.))/30, C.mean(1,keepdim=True).expand_as(C)/20,
         (C==0).float(), (C==20).float(), (P>0).float(), tt, (P>=10).float(), th.clamp(20-C,min=0)*P/600, oth(mc)/20]
    return th.stack(f, -1)

class Net(nn.Module):
    def __init__(s, F=NF, H=256):
        super().__init__()
        s.m = nn.Sequential(nn.Linear(F, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 21))
    def forward(s, x): return s.m(x)

def simulate(net, rule, C0, T=24, seed=0, ret_hist=False):
    """rule(C, P_hist, t) -> punishment [B,4] (P_hist has rounds 0..t-1). Returns per-game summed mean payoff."""
    g = th.Generator().manual_seed(seed)
    B = C0.shape[0]
    C = C0.float().unsqueeze(-1); P = th.zeros(B, 4, 0)
    pay = th.zeros(B)
    with th.no_grad():
        for t in range(T):
            p = rule(C, P, t).clamp(0, 30).floor()
            P = th.cat([P, p.unsqueeze(-1)], -1)
            c = C[..., t]
            pay = pay + 20 + 0.6 * c.mean(1) - 2 * p.mean(1)
            if t < T - 1:
                x = feats_all(C, P)[:, :, -1]
                pr = th.softmax(net(x), -1)
                cn = th.multinomial(pr.reshape(-1, 21), 1, generator=g).reshape(B, 4).float()
                C = th.cat([C, cn.unsqueeze(-1)], -1)
    return (pay, C, P) if ret_hist else pay
