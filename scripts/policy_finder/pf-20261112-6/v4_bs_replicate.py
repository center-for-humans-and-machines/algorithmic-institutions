"""v4: replicate v3's finding (block-start punishment pays in big groups) on v3's own sweep.
At block start b (4 or 8) the punishment P_b dealt by a v3 point is a known function of the history and the point's
params. E[P_b | history] over the 256 design points is computed for every game; P_b - E[P_b|history] is then
independent of the history (params are drawn independently of the game), so its coefficient on later outcomes is
causal. For b = 8, history includes round-4 punishment, which is a function of the params too: E is over points
with the same round-4 treatment status... simpler: at b=8 I only use points with last_b >= 2 and condition on the
history, which already contains the round-4 punishment (E computed over all points, so a bias is possible; b=4 is
the clean one)."""
import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from load_sweep import load
base = 'plots/simulation/policy_finder/pf-20261112-6-v3_sweep/'
if not os.path.exists('/tmp/v3sweep.npz'):
    np.savez('/tmp/v3sweep.npz', **load(base))
D = np.load('/tmp/v3sweep.npz')
c, p, g, v, mg, name = D['c'], D['p'], D['g'], D['v'], D['mine_g'], D['name']
pts = {q['name']: q['params'] for q in json.load(open(base + 'sweep.json'))['points']}
keys = sorted(pts)
PAR = np.array([[pts[k]['n_min'], pts[k]['last_b'], pts[k]['thr'], pts[k]['streak'], pts[k]['amt']] for k in keys])
pid = np.searchsorted(np.array(keys), name)
assert (np.array(keys)[pid] == name).all()
N = len(c)
mine = g == mg[:, None, None]
pool = lambda m: (1.6 * c * m - p * m).sum(1)
op, oo = pool(mine), pool(~mine)
size = mine.sum(1)

def would(b, par):
    """punishment total each game would get at round b under params par (vectorised over games)."""
    nmin, lb, thr, st, amt = par
    if b > 4 * lb: return np.zeros(N)
    low = np.ones((N, 8), bool)
    for k in range(int(st)):
        low &= c[:, :, b - k] <= thr
    here = mine[:, :, b] & (v[:, :, b] > 0)
    n = (low & here).sum(1)
    return np.floor(amt) * n * (size[:, b] >= nmin)

for b in [4, 8]:
    Pb = (p[:, :, b] * mine[:, :, b]).sum(1)
    W = np.stack([would(b, PAR[i]) for i in range(len(keys))], 1)  # [N, 256]
    chk = W[np.arange(N), pid]
    print(f'b={b}: check would==actual {np.mean(np.abs(chk - Pb) < 1e-6):.4f}; mean P {Pb.mean():.2f}')
    E = W.mean(1)
    if b == 8:  # condition on round-4 treatment: use only points that punish at 8, E over them
        ok = PAR[pid, 1] >= 2
        E = W[:, PAR[:, 1] >= 2].mean(1)
    else:
        ok = np.ones(N, bool)
    R = Pb - E
    rest_lead = (op[:, b:] - oo[:, b:]).sum(1) / 24  # contribution to the game score
    outs = {'lead(score units)': rest_lead}
    for r in [b + 4, b + 8, 20]:
        if r < 24: outs[f'size r{r}'] = size[:, r].astype(float)
    outs['own c/round rest'] = (c * mine)[:, :, b + 1:].sum(1).mean(1)
    outs['oth c/round rest'] = (c * ~mine)[:, :, b + 1:].sum(1).mean(1)
    outs['oth c/member b+1..b+3'] = ((c * ~mine)[:, :, b + 1:b + 4].sum(1) / np.maximum((~mine)[:, :, b + 1:b + 4].sum(1), 1)).mean(1)
    for lab, sel in [('all', ok), ('n<=4', ok & (size[:, b] <= 4)), ('n=5', ok & (size[:, b] == 5)),
                     ('n=6', ok & (size[:, b] == 6)), ('n>=7', ok & (size[:, b] >= 7))]:
        if R[sel].std() < 1e-6: continue
        line = f'  {lab:5s} games {sel.sum():6d} sd(R) {R[sel].std():5.2f} |'
        for k, y in outs.items():
            X = np.column_stack([np.ones(sel.sum()), R[sel], E[sel]])
            bb, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
            e = y[sel] - X @ bb
            se = np.sqrt(np.linalg.inv(X.T @ X)[1, 1] * (e ** 2).mean())
            line += f' {k} {bb[1]:+.4f}({se:.4f})'
        print(line)
