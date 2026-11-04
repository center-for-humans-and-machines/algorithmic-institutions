"""Coarser version of v7_cells: aggregate the per-cell intensities into few features to
cut collinearity: (block 0, blocks 1-2, blocks 3-4) x (giver 0-4, 5-9, 10+), positions 0-1 summed."""
import numpy as np
z = np.load('/tmp/v7_cells.npz'); F, Y, V = z['F'], z['Y'], z['V']
names = list(z['names'])
def col(n): return F[:, names.index(n)]
groups = {}
for cb in ['0-4', '5-9', '10-20']:
    groups[f'blk0_{cb}'] = col(f'b0p0_{cb}') + col(f'b0p1_{cb}')
    groups[f'blk12_{cb}'] = col(f'b1p0_{cb}') + col(f'b1p1_{cb}') + col(f'b2p0_{cb}') + col(f'b2p1_{cb}')
    groups[f'blk34_{cb}'] = col(f'b34p0_{cb}') + col(f'b34p1_{cb}')
G = np.column_stack(list(groups.values())); gn = list(groups)
print('corr matrix'); print(np.round(np.corrcoef(G.T), 2))
Gs = (G - G.mean(0)) / G.std(0)
Z = np.column_stack([np.ones(len(Y)), Gs] + [(V == v).astype(float) for v in ['v4', 'v5', 'v6']])
b = np.linalg.lstsq(Z, Y, rcond=None)[0]; r = Y - Z @ b
se = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * r.var())
print('R2', 1 - r.var() / Y.var())
for i, n in enumerate(gn):
    print(f'{n:14s} mean {G[:, i].mean():6.1f} sd {G[:, i].std():5.1f} per sd {b[1+i]:+.2f} ({se[1+i]:.2f}) per point {b[1+i]/G[:, i].std():+.3f}')
# quadratic in the three most important
Z2 = np.column_stack([Z, Gs ** 2])
b2 = np.linalg.lstsq(Z2, Y, rcond=None)[0]; r2 = Y - Z2 @ b2
print('R2 quad', 1 - r2.var() / Y.var())
for i, n in enumerate(gn):
    lin, q = b2[1+i], b2[1+len(gn)+3+i]
    pk = -lin / (2*q) if q < 0 else np.nan
    print(f'{n:14s} lin {lin:+.2f} quad {q:+.2f} peak at {G[:, i].mean()+pk*G[:, i].std():.1f}' )
print('\nper version, linear, features: block0 / blocks1-2 / blocks3-4 dose to givers 0-9, and block0 10+')
H = np.column_stack([groups['blk0_0-4'] + groups['blk0_5-9'], groups['blk12_0-4'] + groups['blk12_5-9'],
                     groups['blk34_0-4'] + groups['blk34_5-9'], groups['blk0_10-20'], groups['blk12_10-20'] + groups['blk34_10-20']])
hn = ['blk0_0-9', 'blk12_0-9', 'blk34_0-9', 'blk0_10+', 'blk1-4_10+']
for v in ['all', 'v3', 'v4', 'v5', 'v6']:
    s = np.ones(len(Y), bool) if v == 'all' else V == v
    Zv = np.column_stack([np.ones(s.sum()), H[s]] + ([(V[s] == w).astype(float) for w in ['v4', 'v5', 'v6']] if v == 'all' else []))
    bv = np.linalg.lstsq(Zv, Y[s], rcond=None)[0]; rv = Y[s] - Zv @ bv
    sev = np.sqrt(np.diag(np.linalg.inv(Zv.T @ Zv)) * rv.var())
    print(v, ' '.join(f'{n}:{bv[1+i]:+.3f}({sev[1+i]:.3f})' for i, n in enumerate(hn)))
