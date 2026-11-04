"""Newcomers to my group (group 0) at switch rounds 4, 8, 12, 16: their contribution
relative to incumbents, and the effect of punishment in their first round on their
later contribution and staying, vs incumbents. Pooled over v2's sweep points."""
import numpy as np
d = np.load('/tmp/pfw/v2.npz'); C, V, P, G = [d[k].astype(np.int16) for k in 'CVPG']
g0 = G == 0
for r0 in (4, 8, 12, 16, 20):
    new = g0[..., r0] & ~g0[..., r0 - 1]; inc = g0[..., r0] & g0[..., r0 - 1]
    # contribution of newcomers vs incumbents at r0, and incumbents' at r0-1
    cn = C[..., r0][new].mean(); ci = C[..., r0][inc].mean()
    # newcomers previous punishment by ah in their last 4 rounds
    pah = P[..., r0 - 4:r0].mean(-1)[new].mean()
    print(f'r{r0}: newcomers {new.sum()} c={cn:.2f} (prev ah pun/round {pah:.2f}); incumbents {inc.sum()} c={ci:.2f}')
    end = min(r0 + 3, 23)
    stay = g0[..., r0:end + 1].all(-1)
    # relative c to others' mean in group at r0
    tot = (C[..., r0] * g0[..., r0]).sum(-1, keepdims=True); n = g0[..., r0].sum(-1, keepdims=True)
    ref = (tot - C[..., r0]) / np.maximum(n - 1, 1)
    ratio = C[..., r0] / np.maximum(ref, 0.5)
    for who, m in (('new', new), ('inc', inc)):
        for rlo, rhi in ((0, .5), (.5, .9), (.9, 9)):
            for plo, phi in ((0, 0), (1, 5), (6, 14), (15, 30)):
                s = m & (ratio >= rlo) & (ratio < rhi) & (P[..., r0] >= plo) & (P[..., r0] <= phi)
                if s.sum() < 200: continue
                ss = s & stay
                dc = (C[..., end] - C[..., r0])[ss].mean()
                print(f'   {who} ratio {rlo}-{rhi} p {plo}-{phi}: n {s.sum():7d} stay3 {stay[s].mean():.2f} dc@+{end-r0} {dc:+.2f}')
