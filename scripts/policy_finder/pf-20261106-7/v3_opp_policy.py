# v3: how each opponent punishes: mean punishment by round and contribution band (own group members)
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv)>1 else '/tmp/v2sweep.npz')
C, P, G, V = d['C'], d['P'], d['G'], d['V']
opp, seat = d['opp'], d['seat']
sel = np.arange(0, len(C), 4)  # subsample
C, P, G, V, opp, seat = C[sel].astype(float), P[sel].astype(float), G[sel], V[sel], opp[sel], seat[sel]
theirs = (G != seat[:, None, None]) & (V == 1)
bands = [(0,4),(5,9),(10,14),(15,20)]
for o in range(1,5):
    m = opp == o
    print('== opponent', o)
    for r in [0,1,2,3,4,5,8,9,12,13,16,20]:
        row = []
        for lo,hi in bands:
            mk = theirs[m][:,:,r] & (C[m][:,:,r]>=lo) & (C[m][:,:,r]<=hi)
            row.append(P[m][:,:,r][mk].mean() if mk.sum() else np.nan)
        # also relation to group mean
        print(r, np.round(row,1))
