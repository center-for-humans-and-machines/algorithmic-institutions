"""v3: first screen of structural ideas against v2's best, on both surrogates."""
import sys; sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from screen import *

show('zero', ev(lambda c, v, p, t: zeros(c)))
show('v2 best', ev(v2()))
for S in [4, 6, 8, 10, 12, 23]:
    show(f'v2 S={S}', ev(v2(S=S)))

# 1. scale round 0 punishment
def r0scale(k, **kw):
    base = v2(**kw)
    def f(c, v, p, t):
        x = base(c, v, p, t)
        return x * k if t == 0 else x
    return f
for k in [0, 0.5, 1.5]:
    show(f'round0 x{k}', ev(r0scale(k)))

# 2. late maintenance: after S punish c<=F by Q
def maint(F, Q, S=8, E=2):
    base = v2(S=S)
    def f(c, v, p, t):
        if t <= S:
            return base(c, v, p, t)
        if t > 23 - E:
            return zeros(c)
        return th.where(c[..., -1] <= F, Q + zeros(c), zeros(c))
    return f
for F in [4, 9]:
    for Q in [3, 10, 25]:
        show(f'maint F={F} Q={Q}', ev(maint(F, Q)))

# 3. ratchet tap: after S, tap players whose c dropped by >= d vs own previous
def ratchet(d, Q, S=8, E=2):
    base = v2(S=S)
    def f(c, v, p, t):
        if t <= S:
            return base(c, v, p, t)
        if t > 23 - E:
            return zeros(c)
        drop = c[..., -2] - c[..., -1]
        return th.where((drop >= d) & (v[..., -2] > 0), Q + zeros(c), zeros(c))
    return f
for d in [1, 3, 6]:
    for Q in [1, 3, 8]:
        show(f'ratchet d={d} Q={Q}', ev(ratchet(d, Q)))

# 4. adaptive: punish (v2 schedule) only while group mean of valid c below M, any round <= 23-E
def adaptive(M, E=3, **kw):
    base = v2(S=23, **kw)
    def f(c, v, p, t):
        if t > 23 - E:
            return zeros(c)
        cc = c[..., -1]; vv = v[..., -1]
        gm = (cc * vv).sum(1, keepdim=True) / vv.sum(1, keepdim=True).clamp(min=1)
        return th.where(gm < M, base(c, v, p, t), zeros(c))
    return f
for M in [12, 14, 16, 18]:
    for E in [3, 8]:
        show(f'adaptive M={M} E={E}', ev(adaptive(M, E)))
