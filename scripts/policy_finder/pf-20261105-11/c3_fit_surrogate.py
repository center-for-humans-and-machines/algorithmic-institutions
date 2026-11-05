"""v3: fit the surrogate on v1+v2 sweeps + reference sims, holding out 25 runs per sweep."""
import numpy as np, torch as th, sys
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from surrogate import Net, rows
th.manual_seed(0)
Xs, Ys, Ms, init = [], [], [], []
held = {}
for tag in ['v1', 'v2', 'ref']:
    z = np.load(f'/tmp/pf3/{tag}.npz', allow_pickle=True)
    runs = z['run']
    ur = np.unique(runs)
    hold = set(ur[::10]) if tag != 'ref' else set()
    held[tag] = sorted(hold)
    keep = ~np.isin(runs, list(hold))
    # subsample episodes: 100 per sweep run, all reference
    sel = np.where(keep)[0]
    if tag != 'ref':
        sel = sel[(np.arange(len(sel)) % 5) == 0]
    c = th.tensor(z['c'][sel]).float(); p = th.tensor(z['p'][sel]).float(); v = th.tensor(z['v'][sel]).float()
    X, Y, M = rows(c, p, v)
    Xs.append(X[M]); Ys.append(Y[M])
    vv = z['v'][:, :, 0].all(1)
    init.append(th.tensor(z['c'][:, :, 0][vv]))
    print(tag, len(sel), M.sum().item())
X = th.cat(Xs); Y = th.cat(Ys)
init_c = th.cat(init)
z = np.load('/tmp/pf3/v1.npz')
to_rate = th.tensor(1 - z['v'].mean((0, 1))).float()
print('rows', len(X), 'timeout rate by round', to_rate.numpy().round(3))
perm = th.randperm(len(X)); nval = 200000
Xv, Yv = X[perm[:nval]], Y[perm[:nval]]; Xt, Yt = X[perm[nval:]], Y[perm[nval:]]
net = Net()
opt = th.optim.Adam(net.parameters(), lr=2e-3)
th.set_num_threads(2)
for ep in range(int(sys.argv[1]) if len(sys.argv) > 1 else 4):
    pr = th.randperm(len(Xt))
    for i in range(0, len(Xt), 4096):
        b = pr[i:i + 4096]
        loss = th.nn.functional.cross_entropy(net(Xt[b]), Yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    for gr in opt.param_groups: gr['lr'] *= 0.5
    with th.no_grad():
        print(ep, 'val nll', th.nn.functional.cross_entropy(net(Xv), Yv).item(), flush=True)
th.save(dict(net=net.state_dict(), init_c=init_c, to_rate=to_rate, held=held), '/tmp/pf3/surrogate.pt')
