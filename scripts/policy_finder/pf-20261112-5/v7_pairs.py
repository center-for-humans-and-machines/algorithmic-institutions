"""v7: game pairs across the six sweeps (aligned by slot, seat, episode) that are identical until round R and
differ in my punishment at R. Saves per pair: R, dose diff at R (a-b), state at R (my size, other size, newcomers,
my mean c, n targets), later points diff by phase, lead/own/oth/mem diffs over R..23. -> /tmp/pf/pairs.npz"""
import numpy as np, itertools
V = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
def load(v):
    z = np.load(f'/tmp/pf/{v}.npz')
    key = z['point'].astype(int) * 2 + z['seat'].astype(int)
    o = np.argsort(key, kind='stable')
    return {k: z[k][o] for k in ['c', 'p', 'mine', 'v']}, key[o]
D = {}
for v in V: D[v], k = load(v)
out = {k: [] for k in ['R', 'dd', 'n', 'no', 'new', 'cm', 'ntar', 'late', 'dlead', 'down', 'doth', 'dmem', 'pa', 'pb', 'va', 'vb', 'slot', 'na', 'nb', 'nlow', 'nlow2']}
for a, b in itertools.combinations(V, 2):
    A, B = D[a], D[b]
    pa = (A['p'] * A['mine']).astype(np.int16); pb = (B['p'] * B['mine']).astype(np.int16)
    diff = (pa != pb).any(1)  # [G,24]
    has = diff.any(1); R = np.where(has, diff.argmax(1), -1)
    for r in [0, 1, 4, 5, 8, 12]:
        i = np.where(R == r)[0]
        if len(i) == 0: continue
        m = A['mine'][i].astype(bool); c = A['c'][i].astype(float)
        poolA = 1.6 * A['c'][i] * A['v'][i] - A['p'][i]; poolB = 1.6 * B['c'][i] * B['v'][i] - B['p'][i]
        mA = A['mine'][i].astype(bool); mB = B['mine'][i].astype(bool)
        leadA = ((poolA * mA).sum(1) - (poolA * ~mA).sum(1))[:, r:].mean(1)
        leadB = ((poolB * mB).sum(1) - (poolB * ~mB).sum(1))[:, r:].mean(1)
        ownA = (poolA * mA).sum(1)[:, r:].mean(1); ownB = (poolB * mB).sum(1)[:, r:].mean(1)
        out['R'].append(np.full(len(i), r)); out['dd'].append((pa[i, :, r] - pb[i, :, r]).sum(1))
        out['pa'].append(pa[i, :, r].sum(1)); out['pb'].append(pb[i, :, r].sum(1))
        out['na'].append((pa[i, :, r] > 0).sum(1)); out['nb'].append((pb[i, :, r] > 0).sum(1))
        out['nlow'].append(((c[:, :, r] <= 5) & m[:, :, r]).sum(1)); out['nlow2'].append(((c[:, :, r] <= 2) & m[:, :, r]).sum(1))
        out['n'].append(m[:, :, r].sum(1)); out['no'].append((~m[:, :, r]).sum(1))
        prev = m[:, :, r - 1] if r > 0 else np.zeros_like(m[:, :, 0])
        out['new'].append((m[:, :, r] & ~prev).sum(1))
        out['cm'].append((c[:, :, r] * m[:, :, r]).sum(1) / np.maximum(m[:, :, r].sum(1), 1))
        out['ntar'].append(((pa[i, :, r] - pb[i, :, r]) != 0).sum(1))
        L = np.zeros((len(i), 3))
        for j, (lo, hi) in enumerate([(r + 1, 8), (8, 12), (12, 24)]):
            if hi > r + 1 or lo > r: L[:, j] = (pa[i, :, max(lo, r + 1):hi] - pb[i, :, max(lo, r + 1):hi]).sum((1, 2)) if hi > max(lo, r + 1) else 0
        out['late'].append(L)
        out['dlead'].append(leadA - leadB); out['down'].append(ownA - ownB)
        out['doth'].append(leadA - ownA - (leadB - ownB))
        out['dmem'].append(mA.sum(1)[:, r:].mean(1) - mB.sum(1)[:, r:].mean(1))
        out['va'].append(np.full(len(i), V.index(a))); out['vb'].append(np.full(len(i), V.index(b)))
        out['slot'].append(k[i] // 2)
    print(a, b, {r: int((R == r).sum()) for r in [0, 1, 4, 5, 8, 12]}, flush=True)
np.savez('/tmp/pf/pairs.npz', **{k: np.concatenate(v) for k, v in out.items()})
