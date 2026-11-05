"""v6: one row per game of the v2/v4/v5 real sweeps (same rule family): the rule's params (T, A, B, S), the
round-0 lineup (valid players' sorted c0, mean, min, n valid), and the game's score. -> /tmp/pf6/groups.npz"""
import numpy as np, json
B = 'plots/simulation/policy_finder/'
rows = []
for tag, ver in [('v2', 'pf-20261105-11-v2'), ('v4', 'pf-20261105-11-v4'), ('v5', 'pf-20261105-11-v5')]:
    sw = json.load(open(f'{B}{ver}_sweep/sweep.json'))
    par = {pt['name']: pt['params'] for pt in sw['points']}
    z = np.load(f'/tmp/pf6/{tag}.npz', allow_pickle=True)
    c = z['c'].astype(float); p = z['p'].astype(float); v = z['v'].astype(float)
    c = c * v
    n = v.sum(1)
    pay = 20 - c - p + ((1.6 * c.sum(1) - p.sum(1)) / np.maximum(n, 1))[:, None, :]
    score = ((pay * v).sum(1) / np.maximum(n, 1) * (n > 0)).sum(1)
    names = [r.split('managed by ')[1] for r in z['run']]
    P = np.array([[par[nm]['T'], par[nm]['A'], par[nm]['B'], par[nm]['S']] for nm in names])
    c0 = np.where(v[:, :, 0] > 0, c[:, :, 0], np.nan)
    srt = np.sort(np.where(np.isnan(c0), 99, c0), 1); srt[srt == 99] = np.nan
    mean0 = np.nanmean(c0, 1); n0 = v[:, :, 0].sum(1)
    bill = p.sum((1, 2))
    rows.append(np.column_stack([P, srt, mean0, n0, score, bill, np.full(len(P), int(tag[1]))]))
R = np.concatenate(rows)
np.savez('/tmp/pf6/groups.npz', R=R, cols=np.array(['T', 'A', 'B', 'S', 's0', 's1', 's2', 's3', 'mean0', 'n0', 'score', 'bill', 'ver']))
print(R.shape, np.isnan(R[:, 8]).sum())
