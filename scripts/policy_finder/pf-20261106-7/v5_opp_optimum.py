# v5: per opponent, where is the pool's optimum in my mean punishment per member (and its timing)? all v1-v4 points
import numpy as np, pandas as pd
df = pd.read_csv('/tmp/v5_points.csv')
for o in range(1, 5):
    s = df[df.opp == o]
    X = np.c_[np.ones(len(s)), s.ppm, s.ppm**2]
    b = np.linalg.lstsq(X, s.pool, rcond=None)[0]
    print('opp', o, 'pool = %.1f + %.2f ppm %.2f ppm^2 -> best ppm %.2f' % (b[0], b[1], b[2], -b[1] / (2 * b[2])))
    for lo, hi in [(0, .3), (.3, .6), (.6, .9), (.9, 1.2), (1.2, 1.6), (1.6, 2.2), (2.2, 3), (3, 9)]:
        m = (s.ppm >= lo) & (s.ppm < hi)
        print('   ppm %.1f-%.1f n=%3d pool %.1f members %.2f cpm %.2f' % (lo, hi, m.sum(), s.pool[m].mean(), s.n[m].mean(), s.cpm[m].mean()))
w = df.pivot_table(index=['v', 'pt'], columns='opp', values='pool')
print(w.corr().round(2))
