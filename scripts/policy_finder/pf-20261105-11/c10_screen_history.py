"""v3: second screen: rules that use each player's history or the group context."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *
T0, A0, B0, S0 = 14.3, 1.7, 1.56, 8
show('v2 best', ev(v2()))


def base_amt(cc, T=T0, A=A0, B=B0):
    s = th.relu(T - cc)
    return th.where(s > 0, A + B * s, th.zeros_like(cc))


# (c) improvement credit: if c rose vs last round, multiply by q
def credit(q, S=S0, T=T0, B=B0, A=A0):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        x = base_amt(c[..., -1], T, A, B)
        if t == 0: return x
        up = c[..., -1] > c[..., -2]
        return th.where(up, x * q, x)
    return f


for q in [0, 0.3, 0.6, 1.5]:
    show(f'credit q={q}', ev(credit(q)))
for q, S, T, B in [(0, 10, 15, 1.6), (0.3, 10, 15, 2.0), (0, 12, 16, 2.0), (0.5, 10, 16, 2.0)]:
    show(f'credit q={q} S={S} T={T} B={B}', ev(credit(q, S, T, B)))


# (b) give up: no punishment for a player already punished n times
def giveup(n, S=S0):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        x = base_amt(c[..., -1])
        if t == 0: return x
        k = (p > 0).float().sum(-1)
        return th.where(k >= n, zeros(c), x)
    return f


for n in [1, 2, 3, 5]:
    show(f'giveup n={n}', ev(giveup(n)))


def others_mean(c, v):
    cc = c[..., -1]; vv = v[..., -1]
    no = (vv.sum(1, keepdim=True) - vv).clamp(min=1)
    return ((cc * vv).sum(1, keepdim=True) - cc * vv) / no


# (a) relative target: T_i = max(T, others' mean - d)
def reltarget(d, S=S0, lo=T0):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        T = th.maximum(others_mean(c, v) - d, th.full_like(c[..., -1], lo))
        return base_amt(c[..., -1], T)
    return f


for d in [0, 2, 4]:
    show(f'reltarget max(T, om-{d})', ev(reltarget(d)))


# (a2) target = min(T, others' mean + d)
def reltarget2(d, S=S0, hi=T0):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        T = th.minimum(others_mean(c, v) + d, th.full_like(c[..., -1], hi))
        return base_amt(c[..., -1], T)
    return f


for d in [0, 3, 6]:
    show(f'reltarget min(T, om+{d})', ev(reltarget2(d)))


# (e) only the lowest contributor(s) of the group
def lowest(S=S0, T=T0, A=A0, B=B0):
    def f(c, v, p, t):
        if t > S: return zeros(c)
        cc = c[..., -1] + (1 - v[..., -1]) * 99
        m = cc.amin(1, keepdim=True)
        return th.where(cc <= m, base_amt(cc, T, A, B), zeros(c))
    return f


for S, T, B in [(8, 14.3, 1.56), (12, 16, 2.0), (23, 16, 2.0)]:
    show(f'lowest only S={S} T={T} B={B}', ev(lowest(S, T, B=B)))
