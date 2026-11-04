"""Check the cached arrays: recompute each point's pool (group 0) and compare with sweep.json."""
import sys, numpy as np
z = np.load(sys.argv[1]); c, p, g = z['c'].astype(float), z['p'].astype(float), z['g']
m = (g == 0)
pool = ((1.6*c - p) * m).sum(2).mean((1, 2))
print(np.corrcoef(pool, z['pool'])[0, 1], np.abs(pool - z['pool']).max())
print(np.c_[pool[:5], z['pool'][:5]])
