"""v4: the switch model sees each group's common_good = (1.6*sum c - sum p)/n per round.
At each decision d (after round 3,7,..): OLS of the rule group's net member gain on both
groups' per-member common good in each of the 4 rounds of the block (plus size), episode level."""
import sys, numpy as np
for f in sys.argv[1:]:
    z = np.load(f); c, p, g, v = (z[k].astype(np.float32) for k in 'cpgv')
    c = c.reshape(-1, 8, 24); p = p.reshape(-1, 8, 24); g = g.reshape(-1, 8, 24); v = v.reshape(-1, 8, 24)
    def cg(grp):
        m = (g == grp) * v
        n = m.sum(1)
        return np.where(n > 0, ((1.6*c - p)*m).sum(1)/np.maximum(n, 1), 0), (g == grp).sum(1)
    cg0, n0 = cg(0); cg1, n1 = cg(1)
    for d in (3, 7, 11):
        rr = list(range(d-3, d+1))
        y = n0[:, d+1] - n0[:, d]
        X = np.column_stack([np.ones(len(y)), cg0[:, rr], cg1[:, rr], n0[:, d]])
        ok = (n0[:, d] > 0) & (n1[:, d] > 0)
        b = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        print(f.split('/')[-1], 'dec', d, 'own cg r-3..r0', b[1:5].round(3), 'other', b[5:9].round(3), 'n', b[9].round(2))
