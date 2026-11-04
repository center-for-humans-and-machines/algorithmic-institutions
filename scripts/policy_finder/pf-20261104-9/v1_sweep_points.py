"""Per design point of v1's sweep: pool, members, contribution and punishment per member,
by phase; check the rebuilt pool against sweep.json; correlate with pool."""
import numpy as np, json
z = np.load('/tmp/v1sweep.npz'); C, V, P, G = [z[k].astype(np.float32) for k in 'CVPG']
s = json.load(open('/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9_sweep/sweep.json'))
pts = {int(p['name'].split('_s')[-1]): p for p in s['points']}
mine = (G == 0)
pool = (1.6 * C * mine - P * mine).sum(2)            # [pt, ep, round]
n = mine.sum(2)
cm = (C * mine).sum(2) / np.maximum(n, 1); pm = (P * mine).sum(2) / np.maximum(n, 1)
po = (1.6 * C * ~mine - P * ~mine).sum(2)
ref = np.array([pts[i]['pool'] for i in range(256)])
print('rebuilt vs sweep pool: corr %.4f, mean diff %.3f' % (np.corrcoef(pool.mean((1, 2)), ref)[0, 1], (pool.mean((1, 2)) - ref).mean()))
ph = lambda x: x.reshape(256, -1, 6, 4).mean((1, 3))
P_, N_, C_, Pm_ = ph(pool), ph(n), ph(cm), ph(pm)
order = np.argsort(-ref)
print('pt pool | members by phase | c/member by phase | p/member by phase | other pool')
for i in list(order[:10]) + list(order[-5:]):
    print(i, round(ref[i], 1), N_[i].round(2), C_[i].round(1), Pm_[i].round(2), round(po[i].mean(), 1))
tot_p = pm.mean((1, 2)); tot_n = n.mean((1, 2)); tot_c = cm.mean((1, 2))
for nm, x in [('members', tot_n), ('c/member', tot_c), ('p/member', tot_p)]:
    print(nm, 'corr with pool %.2f' % np.corrcoef(x, ref)[0, 1])
# regress pool on members and c per member and p per member
X = np.c_[np.ones(256), tot_n, tot_c, tot_p]
b = np.linalg.lstsq(X, ref, rcond=None)[0]; print('pool ~ members, c, p:', b.round(2))
# bins of total punishment per member
for lo, hi in [(0, .3), (.3, .6), (.6, 1), (1, 1.5), (1.5, 2.5), (2.5, 10)]:
    m = (tot_p >= lo) & (tot_p < hi)
    if m.sum(): print('p/member %.1f-%.1f: n=%d pool %.1f members %.2f c %.2f' % (lo, hi, m.sum(), ref[m].mean(), tot_n[m].mean(), tot_c[m].mean()))
