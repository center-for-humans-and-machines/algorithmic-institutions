"""v6: per-point summary over all five sweeps (v1..v5), from /tmp/pf/<v>.npz (compact.py).
Per point: lead, own, oth, mem (mean over games), and punishment points per game by round and by target class
(newcomer = not in my group the round before; c band <=5 / >5). Saves /tmp/pf/points.npz."""
import numpy as np
out = {}
for v in ['v1', 'v2', 'v3', 'v4', 'v5']:
    zz = np.load(f'/tmp/pf/{v}.npz')
    c = zz['c'].astype(np.float32); p = zz['p'].astype(np.float32); m = zz['mine'].astype(bool)
    val = zz['v'].astype(np.float32); pt = zz['point']
    pool = 1.6 * c * val - p
    lead = ((pool * m).sum(1) - (pool * ~m).sum(1)).mean(1)
    own = (pool * m).sum(1).mean(1); oth = (pool * ~m).sum(1).mean(1); mem = m.sum(1).mean(1)
    prevm = np.concatenate([m[:, :, :1], m[:, :, :-1]], 2)
    pm = p * m
    new = pm * (~prevm); stay = pm * prevm
    lo = c <= 5
    # per game: [round, class] class: new&lo, new&hi, stay&lo, stay&hi
    F = np.stack([(new * lo).sum(1), (new * ~lo).sum(1), (stay * lo).sum(1), (stay * ~lo).sum(1)], 2)  # [G, 24, 4]
    # contribution of my members per round, members per round
    cm = (c * m).sum(1) / np.maximum(m.sum(1), 1)
    pts = np.unique(pt)
    agg = lambda a: np.stack([a[pt == k].mean(0) for k in pts])
    out[f'{v}_slot'] = pts
    out[f'{v}_F'] = agg(F)
    for k, a in dict(lead=lead, own=own, oth=oth, mem=mem).items():
        out[f'{v}_{k}'] = agg(a)
    out[f'{v}_memr'] = agg(m.sum(1).astype(np.float32)); out[f'{v}_cm'] = agg(cm)
    out[f'{v}_params'] = zz['params']; out[f'{v}_pidx'] = zz['param_idx']
    print(v, len(pts), 'lead %.2f' % out[f'{v}_lead'].mean())
    del c, p, m, val, pool, zz
np.savez('/tmp/pf/points.npz', **out)
