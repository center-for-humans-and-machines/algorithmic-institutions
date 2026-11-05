# v7: at block starts (4, 8, 12) in the final gp games: contributions of stayers vs arrivals in my group,
# at t-1 and t, and their drop; also by whether the player was punished in the previous block
import numpy as np, sys
d = np.load(sys.argv[1] if len(sys.argv) > 1 else '/tmp/v6final.npz')
sel = d['point'] == -1
C, P, G = [d[k][sel].astype(float) for k in 'CPG']
opp, seat = d['opp'][sel], d['seat'][sel]
mine = (G == seat[:, None, None])
for o in np.unique(opp):
    s = opp == o
    for t in [4, 8, 12, 16]:
        m0, m1 = mine[s][:, :, t - 1], mine[s][:, :, t]
        c = C[s]
        stay = m0 & m1; arr = (~m0) & m1; left = m0 & ~m1
        pprev = P[s][:, :, t - 4:t].sum(2)
        out = []
        for nm, g in [('stay', stay), ('arr', arr), ('left', left)]:
            out.append('%s n%.2f c%.1f->%.1f' % (nm, g.sum() / s.sum(), c[:, :, t - 1][g].mean(), c[:, :, t][g].mean()))
        st_p = stay & (pprev > 0); st_np = stay & (pprev == 0)
        out.append('stay punished %.1f->%.1f unpun %.1f->%.1f' % (c[:, :, t - 1][st_p].mean() if st_p.any() else np.nan, c[:, :, t][st_p].mean() if st_p.any() else np.nan, c[:, :, t - 1][st_np].mean(), c[:, :, t][st_np].mean()))
        print('opp', o, 't', t, ' | '.join(out))
