"""v3: screening helper: evaluate python rules on both surrogates (MLP and GRU)."""
import sys, time, torch as th
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
import surrogate, surrogate_gru
th.set_num_threads(2)
SIMS = {'mlp': surrogate.load(), 'gru': surrogate_gru.load()}


def ev(rule, B=4000, seed=1, which=('mlp', 'gru')):
    return {k: SIMS[k].run(rule, B=B, seed=seed) for k in which}


def zeros(c):
    return th.zeros_like(c[..., -1])


def v2(T=14.3, A=1.7, B=1.56, S=8):
    def f(c, v, p, t):
        s = th.relu(T - c[..., -1])
        x = th.where(s > 0, A + B * s, zeros(c))
        return x if t <= S else zeros(c)
    return f


def show(name, r):
    print(f"{name:55s} " + "  ".join(f"{k} {x['score']:6.1f}±{x['se']:.1f}" for k, x in r.items()), flush=True)
