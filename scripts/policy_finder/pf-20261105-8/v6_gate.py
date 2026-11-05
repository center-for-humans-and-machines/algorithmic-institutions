"""v6: does the value of one more main-phase round depend on the group's state?

v5 sweep: horizon H in 4..10 is drawn independently of the curve. At round t, groups
under rules with H > t get the full curve (everyone below the standard), groups under
H <= t only the tail (c < 2). Outcome: payoff from round t to the end. Regress on
D = (H > t), its interaction with the group state at t, and controls.
"""
import numpy as np
z = np.load('/tmp/v5sweep.npz'); C = z['C'].astype(float); P = z['P'].astype(float); V = z['V']; R = z['R']; prm = z['prm']
std, rate, dose, H, end = (prm[R, k] for k in range(5))
n = V.sum(1)
pay = (V * (20 - C - P + ((1.6 * (C * V).sum(1) - (P * V).sum(1)) / np.maximum(n, 1))[:, None, :])).sum(1) / np.maximum(n, 1)
Cv = np.where(V, C, np.nan)
for t in range(4, 10):
    Y = pay[:, t:].sum(1)
    m = np.nanmean(Cv[:, :, t], 1); mn = np.nanmin(np.where(V[:, :, t], C[:, :, t], 99), 1)
    m1 = np.nanmean(Cv[:, :, t - 1], 1); cum = P[:, :, :t].sum((1, 2))
    # expected main-phase punishment at t for this group (what D switches on)
    s = std[:, None]; k = (rate * std)[:, None]
    base = np.floor(np.clip(dose[:, None] * (np.clip(s - C[:, :, t], 0, None) / s) ** k, 0, 30)) * V[:, :, t]
    tail = base * (C[:, :, t] < 2)
    extra = (base - tail).sum(1)  # points D adds at round t
    D = (H > t).astype(float)
    ok = ~np.isnan(m) & (H >= t - 1) & (H <= t + 2)  # near the boundary
    X = np.column_stack([np.ones(ok.sum()), D[ok], (D * (m - 14))[ok], m[ok], m1[ok], mn[ok], cum[ok], std[ok], np.log(rate[ok]), dose[ok], end[ok], H[ok]])
    # H itself is collinear with D near boundary; drop H
    X = X[:, :-1]
    b, *_ = np.linalg.lstsq(X, Y[ok], rcond=None)
    res = Y[ok] - X @ b; s2 = res.var(); cov = s2 * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
    # binned effect by group mean
    out = []
    for lo_, hi_ in [(0, 10), (10, 14), (14, 17), (17, 21)]:
        sel = ok & (m >= lo_) & (m < hi_)
        a = Y[sel & (D == 1)].mean() - Y[sel & (D == 0)].mean()
        ex = extra[sel & (D == 1)].mean()
        out.append(f'm{lo_}-{hi_}: n={sel.sum()} dY={a:+.1f} extra={ex:.1f}')
    print(f't={t} D {b[1]:+.2f}({se[1]:.2f}) Dx(m-14) {b[2]:+.2f}({se[2]:.2f}) |', ' '.join(out))
