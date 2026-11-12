"""v6: score candidate round-4 rules on the pooled clean experiment. Fit per cell (value class x own size bin
[x gap bin]) the dose and dose^2/10 effects (as in v6_r4_fits.py), then a rule's predicted rest-of-game lead
over zero = mean over games of sum over its targets of the cell's curve at the rule's dose, with its se."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
D = dict(np.load('/tmp/v6_pool3.npz'))
here = D['here'].astype(bool); c4 = D['c4']; act = D['act']; sz = D['size4'][:, None] * np.ones((1, 8))
gap = D['oth'] - c4
src = D['src']
vcl = [(0, 2), (3, 5), (6, 7), (8, 10)]
scl = [(1, 4), (5, 6), (7, 8)]
use_gap = '--gap' in sys.argv
gcl = [(-99, 6), (6, 99)] if use_gap else [(-99, 99)]
cells = []
for a, z in vcl:
    for sa, sb in scl:
        for ga, gb in gcl:
            cells.append(((a, z, sa, sb, ga, gb), (c4 >= a) & (c4 <= z) & (sz >= sa) & (sz <= sb) & (gap >= ga) & (gap < gb) & here))
R, E = [], []
for _, cm in cells:
    R += [(act * cm).sum(1) - (D['e1'] * cm).sum(1), (act ** 2 / 10 * cm).sum(1) - (D['e2'] * cm).sum(1)]
    E += [(D['e1'] * cm).sum(1), (D['e2'] * cm).sum(1)]
X = np.column_stack([np.ones(len(c4))] + R + E + [src == 1, src == 2, D['size4'], D['seat']]).astype(float)
y = D['lead']; bb, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ bb
V = np.linalg.pinv(X.T @ X) * (e ** 2).mean()

def score(dosefn):
    d = dosefn(c4, sz, gap) * here
    gv = np.zeros(len(bb))
    for i, (_, cm) in enumerate(cells):
        gv[1 + 2 * i] = (d * cm).sum(1).mean(); gv[2 + 2 * i] = (d ** 2 / 10 * cm).sum(1).mean()
    return gv @ bb, np.sqrt(gv @ V @ gv), (d > 0).sum(1).mean()

band = lambda lo, hi, amt: (lambda c, s, g: np.where((c >= lo) & (c <= hi), np.floor(amt), 0.))
cands = {
    'v4 peak 0-4@13 5@2': lambda c, s, g: np.where(c <= 4, 13., np.where(c <= 5, 2., 0.)),
    'v5 peak 2-5@12': band(2, 5, 12.87),
    '3-5@13': band(3, 5, 13), '3-5@18': band(3, 5, 18), '3-5@22': band(3, 5, 22),
    '2-5@18': band(2, 5, 18), '1-5@18': band(1, 5, 18), '0-5@18': band(0, 5, 18), '0-5@22': band(0, 5, 22),
    '0-5@26': band(0, 5, 26), '3-6@18': band(3, 6, 18), '3-7@18': band(3, 7, 18),
    'size: n>=7 0-5@20, else 3-5@15': lambda c, s, g: np.where(s >= 7, np.where(c <= 5, 20., 0.), np.where((c >= 3) & (c <= 5), 15., 0.)),
    'size: n>=5 0-5@20, else 3-5@15': lambda c, s, g: np.where(s >= 5, np.where(c <= 5, 20., 0.), np.where((c >= 3) & (c <= 5), 15., 0.)),
    'size: n>=7 0-5@22, n5-6 3-5@18, n<=4 3-5@13': lambda c, s, g: np.where(s >= 7, np.where(c <= 5, 22., 0.), np.where((c >= 3) & (c <= 5), np.where(s >= 5, 18., 13.), 0.)),
    'gap>=3 & c<=5 @18': lambda c, s, g: np.where((c <= 5) & (g >= 3), 18., 0.),
    'gap>=4 & c<=5 @18': lambda c, s, g: np.where((c <= 5) & (g >= 4), 18., 0.),
    'gap>=6 & c<=5 @18': lambda c, s, g: np.where((c <= 5) & (g >= 6), 18., 0.),
    'gap>=6 & c<=5 @22': lambda c, s, g: np.where((c <= 5) & (g >= 6), 22., 0.),
    'gap>=6 & c<=7 @18': lambda c, s, g: np.where((c <= 7) & (g >= 6), 18., 0.),
    'gap>=6 & c<=7 @22': lambda c, s, g: np.where((c <= 7) & (g >= 6), 22., 0.),
    'gap>=8 & c<=7 @22': lambda c, s, g: np.where((c <= 7) & (g >= 8), 22., 0.),
    'gap>=6 & c 1-5 @20': lambda c, s, g: np.where((c >= 1) & (c <= 5) & (g >= 6), 20., 0.),
}
print('cells', len(cells), 'gap' if use_gap else '')
for k, f in cands.items():
    m, s, n = score(f)
    print(f'  {k:45s} pred lead {m:+.2f} ({s:.2f})  targets/game {n:.2f}')

if '--grid' in sys.argv:
    # region averages: v5's design region vs candidate v6 regions (uniform over ints, amt on a grid)
    def region(los, his, amts):
        vals = [score(band(lo, hi, a))[0] for lo in los for hi in his for a in amts]
        return np.mean(vals), np.min(vals), np.max(vals)
    print('v5 region lo0-4 hi4-7 amt8-28', region(range(0, 5), range(4, 8), np.linspace(8, 28, 9)))
    print('v6 region lo0-2 hi4-6 amt14-28', region(range(0, 3), range(4, 7), np.linspace(14, 28, 8)))
    print('v6 region lo0-2 hi4-6 amt12-28', region(range(0, 3), range(4, 7), np.linspace(12, 28, 9)))
    print('v6 region lo0-2 hi5-5 amt14-28', region(range(0, 3), [5], np.linspace(14, 28, 8)))
    for lo in range(0, 4):
        print(' lo', lo, [round(score(band(lo, hi, 20))[0], 2) for hi in range(4, 8)], '(hi 4..7, amt 20)')
    for a in [10, 14, 18, 22, 26, 29]:
        print(' amt', a, 'band 0-5', round(score(band(0, 5, a))[0], 2), 'band 1-5', round(score(band(1, 5, a))[0], 2))
