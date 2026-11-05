"""v4: a surrogate of the artificial humans, fitted on the three earlier sweeps
(768 rules x 500 games), to compare rule families offline.

features(c, p, v, t) -> per-player features at round t (history 0..t, punishment 0..t)
target: class 0..20 = contribution at t+1, class 21 = timeout at t+1.
Usage: python v4_surrogate.py train   (writes /tmp/v4_surr.pt)
"""
import sys, json, numpy as np, torch as th
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v3_load

VERSIONS = ['pf-20261105-3', 'pf-20261105-3-v2', 'pf-20261105-3-v3']
NF = 30


def features(c, p, v, t):
    """c, v: [B, 4, t+1] (c zeroed where invalid), p: [B, 4, t+1] (incl. round t's
    punishment). Returns [B, 4, NF]."""
    B = c.shape[0]
    f = []
    ct, vt, pt = c[..., t], v[..., t], p[..., t]

    def lag(x, k, fill=0.):
        return x[..., t - k] if t - k >= 0 else th.full_like(x[..., 0], fill)

    vsum = v[..., :t + 1].sum(-1).clamp(min=1)
    own_mean = (c[..., :t + 1] * v[..., :t + 1]).sum(-1) / vsum
    cum_p = p[..., :t + 1].sum(-1)
    n_pun = (p[..., :t + 1] > 0).float().sum(-1)
    max_p = p[..., :t + 1].max(-1).values
    f += [th.full_like(ct, t / 23.), ct / 20, vt, pt / 30,
          lag(c, 1) / 20, lag(v, 1, 1.), lag(p, 1) / 30,
          lag(c, 2) / 20, lag(p, 2) / 30,
          own_mean / 20, cum_p / 100, n_pun / 10, max_p / 30, c[..., 0] / 20,
          ct / 20 - lag(c, 1) / 20]
    # others
    eye = th.eye(4, device=c.device)
    oth = (1 - eye)  # [4,4]
    def others_mean(x, w):
        num = th.einsum('ij,bj->bi', oth, x * w)
        den = th.einsum('ij,bj->bi', oth, w).clamp(min=1)
        return num / den
    o_c = others_mean(ct, vt)
    o_p = th.einsum('ij,bj->bi', oth, pt) / 3
    o_pmax = th.stack([th.cat([pt[:, :i], pt[:, i + 1:]], 1).max(1).values for i in range(4)], 1)
    oc_masked = th.where(vt > 0, ct, th.full_like(ct, float('nan')))
    o_min = th.stack([th.cat([oc_masked[:, :i], oc_masked[:, i + 1:]], 1).nan_to_num(20.).min(1).values for i in range(4)], 1)
    o_max = th.stack([th.cat([oc_masked[:, :i], oc_masked[:, i + 1:]], 1).nan_to_num(0.).max(1).values for i in range(4)], 1)
    o_c1 = others_mean(lag(c, 1), lag(v, 1, 1.))
    o_hist = others_mean(own_mean, th.ones_like(own_mean))
    o_valid = th.einsum('ij,bj->bi', oth, vt) / 3
    o_cum_p = th.einsum('ij,bj->bi', oth, cum_p) / 300
    o_p1 = th.einsum('ij,bj->bi', oth, lag(p, 1)) / 90
    gmean = (ct * vt).sum(1, keepdim=True) / vt.sum(1, keepdim=True).clamp(min=1)
    f += [o_c / 20, o_p / 30, o_pmax / 30, o_min / 20, o_max / 20, o_c1 / 20, o_hist / 20,
          o_valid, o_cum_p, o_p1, (ct - gmean) / 20,
          (ct < o_c).float(), (pt > 0).float(), (o_p > 0).float(), lag(c, 3) / 20]
    out = th.stack(f, -1)
    assert out.shape[-1] == NF, out.shape
    return out


def build(n_ep=60, ep_off=0):
    X, Y = [], []
    for ver in VERSIONS:
        a = v3_load.load(ver)
        c = th.from_numpy(a['c'][:, ep_off:ep_off + n_ep]).reshape(-1, 4, 24)
        p = th.from_numpy(a['p'][:, ep_off:ep_off + n_ep]).reshape(-1, 4, 24)
        v = th.from_numpy(a['v'][:, ep_off:ep_off + n_ep]).reshape(-1, 4, 24).float()
        c = c * v
        for t in range(23):
            X.append(features(c[..., :t + 1], p[..., :t + 1], v[..., :t + 1], t).reshape(-1, NF))
            y = th.where(v[..., t + 1] > 0, c[..., t + 1], th.full_like(c[..., 0], 21.)).long()
            Y.append(y.reshape(-1))
    return th.cat(X), th.cat(Y)


class Net(th.nn.Module):
    def __init__(self, h=192):
        super().__init__()
        self.f = th.nn.Sequential(th.nn.Linear(NF, h), th.nn.SiLU(), th.nn.Linear(h, h), th.nn.SiLU(),
                                  th.nn.Linear(h, h), th.nn.SiLU(), th.nn.Linear(h, 22))
    def forward(self, x):
        return self.f(x)


def train(n_ep=60, epochs=6):
    th.manual_seed(0)
    X, Y = build(n_ep, 0)
    Xv, Yv = build(20, 400)
    print('train', X.shape, 'val', Xv.shape, flush=True)
    net = Net()
    opt = th.optim.Adam(net.parameters(), lr=2e-3)
    bs = 4096
    n = X.shape[0]
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (n // bs + 1))
    for ep in range(epochs):
        perm = th.randperm(n)
        net.train()
        tot = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = th.nn.functional.cross_entropy(net(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        with th.no_grad():
            vl = sum(th.nn.functional.cross_entropy(net(Xv[i:i + 65536]), Yv[i:i + 65536], reduction='sum').item()
                     for i in range(0, Xv.shape[0], 65536)) / Xv.shape[0]
        print(f'epoch {ep} train {tot / n:.4f} val {vl:.4f}', flush=True)
    th.save(net.state_dict(), '/tmp/v4_surr.pt')


if __name__ == '__main__':
    th.set_num_threads(2)
    if sys.argv[1] == 'train':
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 60, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
