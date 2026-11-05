# v1 sweep: pool vs the effective punishment scale k*(1+w*op) (op = opponent's mean punishment per
# head so far, read at round 8), per opponent, for points with small tolerance s0 and quiet 1-2.
import numpy as np, pandas as pd, json
s = json.load(open('plots/simulation/policy_finder/pf-20261106-9_sweep/sweep.json'))
P = pd.DataFrame([dict(pt=int(q['name'].split('_s')[-1]), **q['params']) for q in s['points']]).set_index('pt')
d = np.load('/tmp/v1sweep.npz')
c, p, g, ours, opp, pt = d['c'].astype(np.float32), d['p'].astype(np.float32), d['g'], d['ours'], d['opp'], d['pt']
ing = (g == ours[:, None, None])
n = ing.sum(1)
pool = ((1.6 * c - p) * ing).sum(1)
othp = (p[:, :, :8] * ~ing[:, :, :8]).sum((1, 2)) / np.maximum((~ing[:, :, :8]).sum((1, 2)), 1)
df = pd.DataFrame(dict(pt=pt, opp=opp, pool=pool.mean(1), mem=n.mean(1), op8=othp,
                       clate=((c * ing).sum(1) / np.maximum(n, 1))[:, 16:].mean(1)))
A = df.groupby(['pt', 'opp']).mean().reset_index().join(P, on='pt')
A['scale'] = A.k * (1 + A.w * A.op8)
sel = A[(A.s0 < 3) & (A.quiet.isin([1, 2]))].copy()
sel['sb'] = pd.cut(sel.scale, [0, 0.1, 0.3, 0.6, 1, 2, 4, 8, 30])
print(sel.pivot_table(index='sb', columns='opp', values='pool', aggfunc='mean', observed=True).round(1).to_string())
print(sel.pivot_table(index='sb', columns='opp', values='clate', aggfunc='mean', observed=True).round(1).to_string())
print(sel.pivot_table(index='sb', columns='opp', values='mem', aggfunc='mean', observed=True).round(2).to_string())
print(sel.pivot_table(index='sb', columns='opp', values='pool', aggfunc='size', observed=True).to_string())
