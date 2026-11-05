"""v5: near the v4 plateau, what distinguishes groups that lock in (all >= T by round 3) from those that do not?
Per round: mean c, the lowest valid player's c, the bill; and the round-0 profile."""
import json, numpy as np
ver = 'pf-20261105-11-v4'
z = np.load('/tmp/pf5/v4.npz', allow_pickle=True)
c, p, v = z['c'].astype(float), z['p'].astype(float), z['v'].astype(bool)
pts = {f"{ver}_s{i:03d}": q['params'] for i, q in enumerate(json.load(open(f'plots/simulation/policy_finder/{ver}_sweep/sweep.json'))['points'])}
key = [r.split(' managed by ')[-1] for r in z['run']]
par = {k: np.array([pts[x][k] for x in key]) for k in ['T', 'A', 'B', 'S']}
sel = (par['T'] > 12.5) & (par['T'] < 14.5) & (par['B'] > 1.5) & (par['B'] < 2.3) & (par['S'] >= 6) & (par['S'] <= 10)
T = par['T'][:, None, None]
comply = (~v | (c >= T)).all(1)
grad = comply[:, :4].any(1)
cm = np.where(v, c, np.nan)
print('n sel', sel.sum(), 'grad share', grad[sel].mean().round(2))
for g in [True, False]:
    m = sel & (grad == g)
    print('grad' if g else 'not ')
    print('  mean c ', np.round(np.nanmean(cm[m].mean(1) if False else np.nanmean(cm[m], 1), 0)[[0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 23]], 1))
    print('  min c  ', np.round(np.nanmean(np.nanmin(cm[m], 1), 0)[[0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 23]], 1))
    print('  bill/pl', np.round(p[m].mean((0, 1))[[0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 23]], 2))
    srt = np.sort(np.where(v[:, :, 0], c[:, :, 0], 15), 1)
    print('  sorted c0', np.round(srt[m].mean(0), 1))
# lock-in probability by the round-0 minimum and by the sum of round-0 fines, within c0-min bins
mn0 = np.nanmin(cm[:, :, 0], 1)
P0 = p[:, :, 0].sum(1)
for lo, hi in [(0, 3), (4, 7), (8, 11), (12, 20)]:
    m = sel & (mn0 >= lo) & (mn0 <= hi)
    print(f'min c0 {lo}-{hi}: share {m.sum() / sel.sum():.2f} lock-in {grad[m].mean():.2f}')
