"""v3: fit the GRU surrogate on v1+v2 sweeps (minus held-out runs) + reference sims."""
import numpy as np, torch as th, sys, time
sys.path.insert(0, 'scripts/policy_finder/pf-20261105-11')
from surrogate_gru import GNet, step_input
th.manual_seed(0); th.set_num_threads(2)
CS, PS, VS, init = [], [], [], []
held = {}
for tag in ['v1', 'v2', 'ref']:
    z = np.load(f'/tmp/pf3/{tag}.npz', allow_pickle=True)
    runs = z['run']; ur = np.unique(runs)
    hold = set(ur[::10]) if tag != 'ref' else set()
    held[tag] = sorted(hold)
    sel = np.where(~np.isin(runs, list(hold)))[0]
    if tag != 'ref':
        sel = sel[(np.arange(len(sel)) % int(sys.argv[2])) == 0]
    CS.append(z['c'][sel]); PS.append(z['p'][sel]); VS.append(z['v'][sel])
    vv = z['v'][:, :, 0].all(1); init.append(th.tensor(z['c'][:, :, 0][vv]))
C = th.tensor(np.concatenate(CS)).float(); P = th.tensor(np.concatenate(PS)).float(); V = th.tensor(np.concatenate(VS)).float()
init_c = th.cat(init)
to_rate = th.tensor(1 - np.load('/tmp/pf3/v1.npz')['v'].mean((0, 1))).float()
# precompute inputs [N, 23, 4, NI]
X = th.stack([step_input(C[..., t], P[..., t], V[..., t], t) for t in range(23)], 1)
Y = C[..., 1:].long().permute(0, 2, 1)  # [N,23,4]
M = V[..., 1:].permute(0, 2, 1) > 0
N = len(X); print('episodes', N, flush=True)
perm = th.randperm(N); nv = 3000
vi, ti = perm[:nv], perm[nv:]
net = GNet(64)
opt = th.optim.Adam(net.parameters(), lr=3e-3)

def loss_of(idx):
    x, y, m = X[idx], Y[idx], M[idx]
    hs = net.init(len(idx)); L = 0.
    for t in range(23):
        lo, hs = net.step(x[:, t], hs)
        l = th.nn.functional.cross_entropy(lo.reshape(-1, 21), y[:, t].reshape(-1), reduction='none')
        L = L + (l * m[:, t].reshape(-1)).sum()
    return L / m.sum()

t0 = time.time()
for ep in range(int(sys.argv[1])):
    pr = ti[th.randperm(len(ti))]
    for i in range(0, len(pr), 512):
        loss = loss_of(pr[i:i + 512])
        opt.zero_grad(); loss.backward(); opt.step()
    for gr in opt.param_groups: gr['lr'] *= 0.6
    with th.no_grad():
        print(ep, 'val nll', loss_of(vi).item(), round(time.time() - t0), flush=True)
    th.save(dict(net=net.state_dict(), init_c=init_c, to_rate=to_rate, held=held, h=64), '/tmp/pf3/gru.pt')
