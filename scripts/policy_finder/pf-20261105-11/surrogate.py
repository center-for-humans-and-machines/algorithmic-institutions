"""v3: a surrogate of the artificial humans, fitted on the v1/v2 sweeps and reference sims.
An MLP gives the distribution of c_{t+1} (21 classes) for one player from hand-made features of
the history (own and others' c, p, validity, running means, round). Used to screen rule families
offline, since the real model's weights are not in the container."""
import numpy as np, torch as th

NF = 19


def features(c, p, v, t):
    """c, p, v: float tensors [B, 4, t+1] (rounds 0..t, p includes round t). Returns [B, 4, NF]."""
    ct, pt, vt = c[..., t], p[..., t], v[..., t]
    if t > 0:
        c1, p1, v1 = c[..., t - 1], p[..., t - 1], v[..., t - 1]
    else:
        c1, p1, v1 = ct, th.zeros_like(pt), th.zeros_like(vt)
    cm = c[..., :t + 1].mean(-1)
    pm = p[..., :t + 1].mean(-1)
    pmax = p[..., :t + 1].amax(-1)
    npun = (p[..., :t + 1] > 0).float().mean(-1)
    cmax = c[..., :t + 1].amax(-1)
    n = 3.0
    oc = (ct.sum(1, keepdim=True) - ct) / n
    op = (pt.sum(1, keepdim=True) - pt) / n
    oc1 = (c1.sum(1, keepdim=True) - c1) / n
    ocm = (cm.sum(1, keepdim=True) - cm) / n
    opm = (pm.sum(1, keepdim=True) - pm) / n
    # min of others' c_t
    big = ct.unsqueeze(1).expand(-1, 4, -1).clone()
    eye = th.eye(4, dtype=th.bool).unsqueeze(0).expand(big.shape[0], -1, -1)
    big[eye] = 99.
    ocmin = big.amin(-1)
    tt = th.full_like(ct, t / 23.)
    f = th.stack([ct / 20, pt / 30, vt, c1 / 20, p1 / 30, v1, cm / 20, pm / 30, pmax / 30, npun, cmax / 20,
                  oc / 20, op / 30, oc1 / 20, ocm / 20, opm / 30, ocmin / 20, tt, (ct - c1) / 20], -1)
    return f


class Net(th.nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.m = th.nn.Sequential(th.nn.Linear(NF, h), th.nn.ReLU(), th.nn.Linear(h, h), th.nn.ReLU(),
                                  th.nn.Linear(h, h), th.nn.ReLU(), th.nn.Linear(h, 21))

    def forward(self, x):
        return self.m(x)


def rows(c, p, v, rounds=range(23)):
    X, Y, M = [], [], []
    for t in rounds:
        X.append(features(c[..., :t + 1], p[..., :t + 1], v[..., :t + 1], t))
        Y.append(c[..., t + 1].long())
        M.append(v[..., t + 1] > 0)
    return th.cat(X), th.cat(Y), th.cat(M)


class Sim:
    """Roll out a rule (python fn(c, valid, p, t) -> punishment [B,4]) on the surrogate."""

    def __init__(self, net, init_c, to_rate):
        self.net, self.init_c, self.to_rate = net, init_c, to_rate  # init_c [E,4] round-0 groups (valid)

    @th.no_grad()
    def run(self, rule, B=2000, seed=0):
        g = th.Generator().manual_seed(seed)
        idx = th.randint(len(self.init_c), (B,), generator=g)
        c0 = self.init_c[idx].float()
        perm = th.argsort(th.rand(B, 4, generator=g), 1)
        c0 = th.gather(c0, 1, perm)
        v0 = (th.rand(B, 4, generator=g) > self.to_rate[0]).float()
        C = [c0 * v0 + 15 * (1 - v0)]  # timed-out recorded as 15
        V = [v0]; Pl = []
        pays = []
        for t in range(24):
            c = th.stack(C, -1); v = th.stack(V, -1)
            pp = th.stack(Pl, -1) if Pl else th.zeros(B, 4, 0)
            pun = rule(c * v, v, pp, t)
            pun = th.floor(th.clamp(pun, 0, 30)) * V[-1]
            Pl.append(pun)
            ct, vt = C[-1], V[-1]
            n = vt.sum(1).clamp(min=1)
            pool = (1.6 * (ct * vt).sum(1) - (pun * vt).sum(1)) / n
            pay = 20 - ct - pun + pool[:, None]
            pays.append(((pay * vt).sum(1) / n))
            if t == 23:
                break
            pfull = th.stack(Pl, -1)
            x = features(c, pfull, v, t)
            logits = self.net(x)
            nc = th.distributions.Categorical(logits=logits).sample().float()
            nv = (th.rand(B, 4, generator=g) > self.to_rate[t + 1]).float()
            C.append(nc * nv + 15 * (1 - nv)); V.append(nv)
        pays = th.stack(pays, -1)
        c = th.stack(C, -1); P = th.stack(Pl, -1); v = th.stack(V, -1)
        return dict(score=pays.sum(1).mean().item(), se=pays.sum(1).std().item() / B ** .5,
                    c_round=((c * v).sum((0, 1)) / v.sum((0, 1))).numpy(),
                    p_round=((P * v).sum((0, 1)) / v.sum((0, 1))).numpy(),
                    pay_round=pays.mean(0).numpy())


def load(path='/tmp/pf3/surrogate.pt'):
    s = th.load(path, weights_only=False)
    net = Net(); net.load_state_dict(s['net']); net.eval()
    return Sim(net, s['init_c'], s['to_rate'])
