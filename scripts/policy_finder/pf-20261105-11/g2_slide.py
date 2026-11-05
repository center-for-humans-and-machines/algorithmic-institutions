"""v7: what is the post-stop slide made of? v6 sweep (all S=8) and v4_gp final. Players by c at round 8;
per-player change 8->23; how much of the mean drop comes from collapses vs drift."""
import numpy as np
for tag, filt in [('v4f', 'v4_gp'), ('v6', None)]:
    z = np.load(f'/tmp/pf7/{tag}.npz', allow_pickle=True)
    c, v, run = z['c'].astype(float), z['v'].astype(float), z['run']
    if filt: m = np.array([r.endswith(filt) for r in run]); c, v = c[m], v[m]
    print('==', tag, len(c))
    # use players valid at rounds 8 and 23 (mostly); impute timeouts as nan
    cc = np.where(v > 0, c, np.nan)
    c8 = cc[:, :, 8]; 
    late = np.nanmean(cc[:, :, 19:24], 2)
    d = late - c8
    for lo, hi in [(0, 5), (6, 9), (10, 12), (13, 15), (16, 19), (20, 20)]:
        mm = (c8 >= lo) & (c8 <= hi)
        print(f' c8 {lo:2d}-{hi:2d}: share {np.mean(mm):.3f}  mean c8 {np.nanmean(c8[mm]):5.1f} c19-23 {np.nanmean(late[mm]):5.1f}  contrib to drop {np.nansum(d[mm])/np.sum(~np.isnan(d)):+.2f}')
    print(' drop total', np.nanmean(d))
    # distribution of change
    q = np.nanpercentile(d, [5, 10, 25, 50, 75, 90])
    print(' change pctiles 5,10,25,50,75,90:', np.round(q, 1))
    for thr in [-3, -6, -10]:
        mm = d <= thr; print(f'  change<={thr}: share {np.nanmean(mm):.3f}, contributes {np.nansum(np.where(mm, d, 0))/np.sum(~np.isnan(d)):+.2f}')
    # per round share of players at c<=5, 6-11 among valid
    for r in [8, 10, 12, 15, 18, 21, 23]:
        x = cc[:, :, r]; ok = ~np.isnan(x)
        print(f'  round {r}: c<=5 {np.mean(x[ok]<=5):.3f}  6-11 {np.mean((x[ok]>=6)&(x[ok]<=11)):.3f}  12-16 {np.mean((x[ok]>=12)&(x[ok]<=16)):.3f}  17-20 {np.mean(x[ok]>=17):.3f}  mean {x[ok].mean():.2f}')
