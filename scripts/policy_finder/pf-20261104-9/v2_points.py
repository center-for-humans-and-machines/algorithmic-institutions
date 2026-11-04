"""Per-point summaries of v2's sweep: pool check vs sweep.json, phase profiles of
punishment, contribution per member, members; relation to last_round, fresh."""
import numpy as np, json
d = np.load('/tmp/pfw/v2.npz'); C, V, P, G = [d[k].astype(np.float32) for k in 'CVPG']
sw = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v2_sweep/sweep.json'))
pts = sw['points'] if 'points' in sw else sw
pool_json = np.array([q['pool'] for q in pts]); prm = [q['params'] for q in pts]
idx = np.array([int(q['name'].split('_s')[-1]) for q in pts])
g0 = (G == 0)
pool = (1.6 * (C * g0).sum(2) - (P * g0).sum(2))  # point, ep, round
print('pool check max abs diff', np.abs(pool.mean((1, 2))[idx] - pool_json).max())
mem = g0.sum(2); cpm = (C * g0).sum(2) / np.maximum(mem, 1); ppm = (P * g0).sum(2) / np.maximum(mem, 1)
order = np.argsort(-pool_json)
def ph(x): return x.mean(1).reshape(x.shape[0], 6, 4).mean(2)
PM, CM, MM, PL = ph(ppm), ph(cpm), ph(mem.astype(float)), ph(pool)
print('top 12 and bottom 4 points: params | pool | ppm by phase | c/member by phase | members by phase')
for j in list(order[:12]) + list(order[-4:]):
    i = idx[j]; q = prm[j]
    print(i, {k: round(v, 2) for k, v in q.items()}, round(pool_json[j], 1))
    print('   p', np.round(PM[i], 2), ' c', np.round(CM[i], 1), ' m', np.round(MM[i], 2))
lr = np.array([q['last_round'] for q in prm])
for lo, hi in [(8, 9), (10, 12), (13, 16), (17, 20), (21, 23)]:
    s = (lr >= lo) & (lr <= hi); print('last_round', lo, hi, 'n', s.sum(), 'pool mean', pool_json[s].mean().round(2), 'top-quartile share', np.mean(pool_json[s] > np.quantile(pool_json, .75)).round(2))
# per round contribution per member for top 10 points
print('per-round c/member mean of top 10:', np.round(cpm.mean(1)[idx[order[:10]]].mean(0), 1))
print('per-round p/member mean of top 10:', np.round(ppm.mean(1)[idx[order[:10]]].mean(0), 1))
print('per-round members top 10:', np.round(mem.mean(1)[idx[order[:10]]].mean(0), 2))
print('per-round pool top 10:', np.round(pool.mean(1)[idx[order[:10]]].mean(0), 1))
# ah group
g1 = (G == 1); print('ah pool per round (top10 points):', np.round((1.6*(C*g1).sum(2)-(P*g1).sum(2)).mean(1)[idx[order[:10]]].mean(0), 1))
