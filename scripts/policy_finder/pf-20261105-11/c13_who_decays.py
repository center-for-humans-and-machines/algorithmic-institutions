"""v3: in v2's final best (S=8), who decays after the stop? Late (rounds 18-23) contribution by
round-0 contribution and by the number of rounds the player was punished in the window."""
import numpy as np, pandas as pd
z = np.load('/tmp/pf3/v2f.npz', allow_pickle=True)
m = np.array([r.endswith('_gp') for r in z['run']])
c = z['c'][m].astype(float); p = z['p'][m].astype(float); v = z['v'][m]
c = np.where(v, c, np.nan)
df = pd.DataFrame(dict(c0=c[:, :, 0].ravel(), c8=c[:, :, 8].ravel(), late=np.nanmean(c[:, :, 18:], -1).ravel(),
                       mid=np.nanmean(c[:, :, 9:14], -1).ravel(),
                       npun=(p[:, :, :9] > 0).sum(-1).ravel(), ptot=p[:, :, :9].sum(-1).ravel()))
df['c0b'] = pd.cut(df.c0, [-1, 4, 9, 14, 19, 20])
df['npb'] = pd.cut(df.npun, [-1, 0, 1, 2, 4, 9])
print(df.groupby(['c0b', 'npb'], observed=True).agg(n=('late', 'size'), c8=('c8', 'mean'), mid=('mid', 'mean'), late=('late', 'mean')).round(1).to_string())
print(df.groupby('npb', observed=True)[['c8', 'mid', 'late']].mean().round(2))
# group-level: late c vs number of group members ever punished
