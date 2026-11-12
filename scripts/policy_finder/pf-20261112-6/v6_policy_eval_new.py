"""v6: like v6_policy_eval.py, with cells value class x own size bin x newcomer (joined at the round-3 decision);
scores rules whose round-4 dose depends on the target's c4 and on whether it is a newcomer."""
import numpy as np, sys
D = dict(np.load('/tmp/v6_pool3.npz'))
here = D['here'].astype(bool); c4 = D['c4']; act = D['act']; sz = D['size4'][:, None] * np.ones((1, 8))
new = D['new4'].astype(bool); src = D['src']
vcl = [(0, 2), (3, 5), (6, 7), (8, 10)]
scl = [(1, 4), (5, 6), (7, 8)] if '--nosize' not in sys.argv else [(1, 8)]
cells = []
for a, z in vcl:
    for sa, sb in scl:
        for nw in (False, True):
            cells.append(((a, z, sa, sb, nw), (c4 >= a) & (c4 <= z) & (sz >= sa) & (sz <= sb) & (new == nw) & here))
R, E = [], []
for _, cm in cells:
    R += [(act * cm).sum(1) - (D['e1'] * cm).sum(1), (act ** 2 / 10 * cm).sum(1) - (D['e2'] * cm).sum(1)]
    E += [(D['e1'] * cm).sum(1), (D['e2'] * cm).sum(1)]
X = np.column_stack([np.ones(len(c4))] + R + E + [src == 1, src == 2, D['size4'], D['seat']]).astype(float)
y = D['lead']; bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()

def score(dosefn):
    d = dosefn(c4, new) * here
    gv = np.zeros(len(bb))
    for i, (_, cm) in enumerate(cells):
        gv[1 + 2 * i] = (d * cm).sum(1).mean(); gv[2 + 2 * i] = (d ** 2 / 10 * cm).sum(1).mean()
    return gv @ bb, np.sqrt(gv @ V @ gv), (d > 0).sum(1).mean()

def rule(lo_new, lo_old, hi, a_new, a_old):
    return lambda c, n: np.where(n, np.where((c >= lo_new) & (c <= hi), np.floor(a_new), 0.),
                                 np.where((c >= lo_old) & (c <= hi), np.floor(a_old), 0.))
cands = {
    'v4 peak': lambda c, n: np.where(c <= 4, 13., np.where(c <= 5, 2., 0.)),
    'v5 peak 2-5@12': rule(2, 2, 5, 12.87, 12.87),
    '0-5@22 all': rule(0, 0, 5, 22, 22),
    '1-5@20 all': rule(1, 1, 5, 20, 20),
    'new 0-5@18, old 3-5@18': rule(0, 3, 5, 18, 18),
    'new 0-5@18, old 3-5@14': rule(0, 3, 5, 18, 14),
    'new 0-5@20, old 0-5@20': rule(0, 0, 5, 20, 20),
    'new 0-5@18, old 1-5@18': rule(0, 1, 5, 18, 18),
    'new 0-5@18, none old': rule(0, 99, 5, 18, 0),
    'new 0-5@22, old 3-5@18': rule(0, 3, 5, 22, 18),
    'new 0-5@14, old 3-5@14': rule(0, 3, 5, 14, 14),
    'new 0-6@18, old 3-5@18': lambda c, n: np.where(n, np.where(c <= 6, 18., 0.), np.where((c >= 3) & (c <= 5), 18., 0.)),
    'new 0-7@18, old 3-5@18': lambda c, n: np.where(n, np.where(c <= 7, 18., 0.), np.where((c >= 3) & (c <= 5), 18., 0.)),
    'new 0-5@18, old 3-5@18 + old 0-2@26': lambda c, n: np.where(n, np.where(c <= 5, 18., 0.), np.where((c >= 3) & (c <= 5), 18., np.where(c <= 2, 26., 0.))),
}
print('cells', len(cells))
for k, f in cands.items():
    m, s, n = score(f)
    print(f'  {k:40s} pred lead {m:+.2f} ({s:.2f})  targets/game {n:.2f}')
# paired difference vs v5 peak
base = rule(2, 2, 5, 12.87, 12.87)
def diff(f1, f2):
    d1 = f1(c4, new) * here; d2 = f2(c4, new) * here
    gv = np.zeros(len(bb))
    for i, (_, cm) in enumerate(cells):
        gv[1 + 2 * i] = ((d1 - d2) * cm).sum(1).mean(); gv[2 + 2 * i] = ((d1 ** 2 - d2 ** 2) / 10 * cm).sum(1).mean()
    return gv @ bb, np.sqrt(gv @ V @ gv)
for k in ['0-5@22 all', 'new 0-5@18, old 3-5@18', 'new 0-5@22, old 3-5@18', 'new 0-5@18, old 1-5@18']:
    print(f'  diff vs v5 peak {k:30s} {diff(cands[k], base)[0]:+.2f} ({diff(cands[k], base)[1]:.2f})')
print('  diff new0-5/old3-5@18 vs 0-5@22 all', diff(cands['new 0-5@18, old 3-5@18'], cands['0-5@22 all']))
if '--grid' in sys.argv:
    for lo_old in range(0, 5):
        print(' lo_old', lo_old, [round(score(rule(0, lo_old, 5, an, ao))[0], 2) for an in (14, 18, 22, 26) for ao in (14, 18, 22)],
              '(a_new 14/18/22/26 x a_old 14/18/22)')
    for lo_new in range(0, 4):
        print(' lo_new', lo_new, [round(score(rule(lo_new, 3, hi, 18, 18))[0], 2) for hi in (4, 5, 6, 7)], '(hi 4..7)')
