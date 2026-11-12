"""v6: align the games of v1..v5 sweeps by (slot, seat, episode order) and save the aligned arrays for rounds 0-11
(c, p, mine, valid) to /tmp/pf/al_<v>.npz. Checks that round-0 contributions coincide across versions."""
import numpy as np
V = ['v1', 'v2', 'v3', 'v4', 'v5']
ref = None
for v in V:
    z = np.load(f'/tmp/pf/{v}.npz')
    pt, seat = z['point'].astype(int), z['seat'].astype(int)
    key = pt * 2 + seat
    order = np.argsort(key, kind='stable')
    out = {k: z[k][order] for k in ['c', 'p', 'mine', 'v']}
    out['key'] = key[order]
    np.savez(f'/tmp/pf/al_{v}.npz', **out)
    if ref is None: ref = out
    else:
        same0 = (out['c'][:, :, 0] == ref['c'][:, :, 0]).all(1)
        print(v, 'keys equal', (out['key'] == ref['key']).all(), 'round-0 c identical share %.3f' % same0.mean())
