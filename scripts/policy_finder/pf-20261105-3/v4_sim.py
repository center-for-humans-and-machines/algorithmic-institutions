"""v4: simulate rules on the surrogate (v4_surrogate.py) and check it against the
three earlier sweeps' scores.

python v4_sim.py check [n_games]   -> surrogate score vs sweep pool for all 768 runs
"""
import sys, json, numpy as np, torch as th
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
sys.path.insert(0, '/opt/algorithmic-institutions/src')
import v3_load
from v4_surrogate import Net, features, VERSIONS
from aimanager.manager.rule import read_rule, run_rule

ROOT = '/opt/algorithmic-institutions/'
_net = None
_r0 = None


def net(path='/tmp/v4_surr.pt'):
    global _net
    if _net is None:
        _net = Net(); _net.load_state_dict(th.load(path)); _net.eval()
    return _net


def round0():
    global _r0
    if _r0 is None:
        a = v3_load.load(VERSIONS[1])
        c0 = a['c'][:, :, :, 0].reshape(-1, 4); v0 = a['v'][:, :, :, 0].reshape(-1, 4)
        _r0 = (th.from_numpy(c0 * v0).float(), th.from_numpy(v0).float())
    return _r0


def rule_fn(yml, params):
    rule, code = read_rule(ROOT + yml if not yml.startswith('/') else yml, 'exp1')
    def f(c, v, p, t):
        inp = {'c': c, 'valid': v, 'p': p, 't': th.tensor(float(t))}
        return run_rule(code, params, inp)
    return f


def simulate(fn, n=1000, seed=0, return_traj=False):
    """fn(c, v, p, t) -> raw punishment [n, 4]. Returns score (mean over games of the
    summed per-round mean valid payoff), se, and optionally trajectories."""
    g = th.Generator().manual_seed(seed)
    c0, v0 = round0()
    idx = th.randint(0, c0.shape[0], (n,), generator=g)
    c = c0[idx].unsqueeze(-1).clone(); v = v0[idx].unsqueeze(-1).clone()
    p = th.zeros(n, 4, 0)
    pay = []
    with th.no_grad():
        for t in range(24):
            raw = fn(c, v, p, t)
            pt = th.floor(raw.clamp(0, 30)) * v[..., t]
            p = th.cat([p, pt.unsqueeze(-1)], -1)
            ct, vt = c[..., t], v[..., t]
            nv = vt.sum(1).clamp(min=1)
            share = (1.6 * (ct * vt).sum(1) - pt.sum(1)) / nv
            payoff = 20 - ct - pt + share.unsqueeze(1)
            pay.append((payoff * vt).sum(1) / nv * (vt.sum(1) > 0).float())
            if t < 23:
                logits = net()(features(c, p, v, t))
                y = th.multinomial(th.softmax(logits.reshape(-1, 22), -1), 1, generator=g).reshape(n, 4)
                vn = (y < 21).float()
                cn = th.where(vn > 0, y.float(), th.zeros_like(vn))
                c = th.cat([c, cn.unsqueeze(-1)], -1); v = th.cat([v, vn.unsqueeze(-1)], -1)
    tot = th.stack(pay, 1).sum(1)
    out = (tot.mean().item(), tot.std().item() / n ** 0.5)
    if return_traj:
        return out + (c, p, v, th.stack(pay, 1))
    return out


def check(n=300):
    rows = []
    for ver in VERSIONS:
        a = v3_load.load(ver)
        yml = f'configs/managers/rule_based/{ver}.yml'
        for name, prm, pool in zip(a['runs'], a['params'], a['pool']):
            s, se = simulate(rule_fn(yml, json.loads(prm)), n=n, seed=1)
            rows.append((ver, name, pool, s))
    r = np.array([(x[2], x[3]) for x in rows])
    print('pearson', np.corrcoef(r.T)[0, 1], 'bias', (r[:, 1] - r[:, 0]).mean(), 'rmse', np.sqrt(((r[:, 1] - r[:, 0]) ** 2).mean()))
    for ver in VERSIONS:
        m = np.array([x[0] == ver for x in rows])
        print(ver, 'pearson', np.corrcoef(r[m].T)[0, 1], 'bias', (r[m, 1] - r[m, 0]).mean())
    top = np.argsort(-r[:, 0])[:15]
    for i in top:
        print(rows[i][1], round(r[i, 0], 1), round(r[i, 1], 1))
    json.dump([(x[0], x[1], float(x[2]), float(x[3])) for x in rows], open('/tmp/v4_check.json', 'w'))
    zero = simulate(lambda c, v, p, t: th.zeros_like(c[..., -1]), n=2000)
    print('zero', zero)


if __name__ == '__main__':
    th.set_num_threads(2)
    if sys.argv[1] == 'check':
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
