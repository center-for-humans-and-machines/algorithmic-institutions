"""v3: teacher-forced calibration of the surrogate: predicted vs actual mean c_{t+1} by round."""
import numpy as np, torch as th, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from surrogate import load, features
sim = load()
for tag in ['ref', 'v2f']:
    z = np.load(f'/tmp/pf3/{tag}.npz', allow_pickle=True)
    for r in np.unique(z['run'])[:2]:
        m = z['run'] == r
        c = th.tensor(z['c'][m]).float(); p = th.tensor(z['p'][m]).float(); v = th.tensor(z['v'][m]).float()
        pr, ac = [], []
        with th.no_grad():
            for t in range(23):
                x = features(c[..., :t + 1], p[..., :t + 1], v[..., :t + 1], t)
                pb = th.softmax(sim.net(x), -1)
                e = (pb * th.arange(21.)).sum(-1)
                mm = v[..., t + 1] > 0
                pr.append(e[mm].mean().item()); ac.append(c[..., t + 1][mm].mean().item())
        print(r[-30:], 'pred-actual', np.round(np.array(pr) - np.array(ac), 2))
