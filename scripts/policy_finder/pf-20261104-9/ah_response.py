"""Response to punishment by ratio of own c to others' mean, all managed groups in the reference
sims (ah and zero) and the ah side of v1's sweep: does punishing at/above-average members lift
or drop them in the sim? Change in c after 1 and 3 rounds (stayers)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
z = np.load('/tmp/v1sweep.npz'); C, V, P, G = [z[k].astype(np.float32) for k in 'CVPG']
rows = []
for g in (1,):  # ah's group in v1's sweep
    mem = (G == g) & (V > 0)
    n = mem.sum(2, keepdims=True); tot = (C * mem).sum(2, keepdims=True)
    oth = (tot - C * mem) / np.maximum(n - mem, 1)
    for t in range(0, 20, 2):
        sel = mem[..., t] & (n[..., 0, t][..., None] >= 2) & np.all(G[..., t:t + 4] == g, -1)
        rows.append(pd.DataFrame({'r': (C[..., t] / np.maximum(oth[..., t], .5))[sel], 'o': oth[..., t][sel],
                                  'p': P[..., t][sel], 'd1': (C[..., t + 1] - C[..., t])[sel],
                                  'd3': (C[..., t + 3] - C[..., t])[sel]}))
d = pd.concat(rows)
d['rb'] = pd.cut(d.r, [-.01, .25, .5, .75, 1, 1.25, 1.5, 2, 100]); d['pb'] = pd.cut(d.p, [-1, 0, 2, 5, 9, 14, 30])
pd.set_option('display.width', 200)
print("ah's group in v1's sweep: change after 3 rounds"); print(d.pivot_table('d3', 'rb', 'pb', 'mean').round(2))
print(d.pivot_table('d3', 'rb', 'pb', 'count'))
# absolute level: laggards ratio 0.5-0.75, by others' mean bin
x = d[(d.r > .5) & (d.r <= .75)]
print('ratio 0.5-0.75, change after 3 by others mean x p'); print(x.pivot_table('d3', pd.cut(x.o, [0, 6, 9, 12, 20]), 'pb', 'mean').round(2))
