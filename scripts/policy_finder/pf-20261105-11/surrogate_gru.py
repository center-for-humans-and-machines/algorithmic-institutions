"""v3: GRU surrogate of the artificial humans (the real model is a node+edge RNN, whose weights
are not in the container). Per player, the input at round t is own c, p, valid and the others'
mean/min/max c and mean p; the GRU state predicts c_{t+1} over 21 classes."""
import torch as th

NI = 9


def step_input(ct, pt, vt, t):
    n = vt.sum(1, keepdim=True)
    no = (n - vt).clamp(min=1)
    oc = ((ct * vt).sum(1, keepdim=True) - ct * vt) / no
    op = ((pt * vt).sum(1, keepdim=True) - pt * vt) / no
    B = ct.shape[0]
    big = ct.unsqueeze(1).expand(-1, 4, -1).clone()
    eye = th.eye(4, dtype=th.bool).unsqueeze(0).expand(B, -1, -1)
    lo = big.masked_fill(eye, 99.).amin(-1)
    hi = big.masked_fill(eye, -99.).amax(-1)
    tt = th.full_like(ct, t / 23.)
    return th.stack([ct / 20, pt / 30, vt, oc / 20, op / 30, lo / 20, hi / 20, tt, (n - 4) / 4 + vt * 0], -1)


class GNet(th.nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.h = h
        self.inp = th.nn.Linear(NI, h)
        self.gru = th.nn.GRUCell(h, h)
        self.out = th.nn.Sequential(th.nn.Linear(h + NI, h), th.nn.ReLU(), th.nn.Linear(h, 21))

    def init(self, B):
        return th.zeros(B * 4, self.h)

    def step(self, x, hs):
        B = x.shape[0]
        xf = x.reshape(B * 4, NI)
        hs = self.gru(th.relu(self.inp(xf)), hs)
        return self.out(th.cat([hs, xf], -1)).reshape(B, 4, 21), hs


class Sim:
    def __init__(self, net, init_c, to_rate):
        self.net, self.init_c, self.to_rate = net, init_c, to_rate

    @th.no_grad()
    def run(self, rule, B=2000, seed=0, return_traj=False):
        g = th.Generator().manual_seed(seed)
        idx = th.randint(len(self.init_c), (B,), generator=g)
        c0 = self.init_c[idx].float()
        perm = th.argsort(th.rand(B, 4, generator=g), 1)
        c0 = th.gather(c0, 1, perm)
        v0 = (th.rand(B, 4, generator=g) > self.to_rate[0]).float()
        C = [c0 * v0 + 15 * (1 - v0)]  # timed-out recorded as 15
        V = [v0]; Pl = []; pays = []
        hs = self.net.init(B)
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
            pays.append((pay * vt).sum(1) / n)
            if t == 23:
                break
            logits, hs = self.net.step(step_input(ct, pun, vt, t), hs)
            u = th.rand(B, 4, 1, generator=g)
            cdf = th.softmax(logits, -1).cumsum(-1)
            nc = (cdf < u).sum(-1).clamp(max=20).float()
            nv = (th.rand(B, 4, generator=g) > self.to_rate[t + 1]).float()
            C.append(nc * nv + 15 * (1 - nv)); V.append(nv)
        pays = th.stack(pays, -1)
        c = th.stack(C, -1); P = th.stack(Pl, -1); v = th.stack(V, -1)
        res = dict(score=pays.sum(1).mean().item(), se=pays.sum(1).std().item() / B ** .5,
                   c_round=((c * v).sum((0, 1)) / v.sum((0, 1))).numpy(),
                   p_round=((P * v).sum((0, 1)) / v.sum((0, 1))).numpy(),
                   pay_round=pays.mean(0).numpy())
        if return_traj:
            res.update(c=c, p=P, v=v)
        return res


def load(path='/tmp/pf3/gru.pt'):
    s = th.load(path, weights_only=False)
    net = GNet(s.get('h', 64)); net.load_state_dict(s['net']); net.eval()
    return Sim(net, s['init_c'], s['to_rate'])
