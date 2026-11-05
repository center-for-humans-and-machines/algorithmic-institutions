"""v5: refit v4's surrogate (same features / net) on all four sweeps, v4's included
(its late-phase rules give the surrogate data on punishing late relapses).
Usage: python v5_surrogate.py train [n_ep] [epochs]   (writes /tmp/v5_surr.pt)
"""
import sys, torch as th
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v4_surrogate as S

S.VERSIONS = ['pf-20261105-3', 'pf-20261105-3-v2', 'pf-20261105-3-v3', 'pf-20261105-3-v4']
PATH = '/tmp/v5_surr.pt'


def train(n_ep=60, epochs=8):
    th.manual_seed(0)
    X, Y = S.build(n_ep, 0)
    Xv, Yv = S.build(15, 450)
    print('train', X.shape, 'val', Xv.shape, flush=True)
    net = S.Net()
    opt = th.optim.Adam(net.parameters(), lr=2e-3)
    bs = 4096; n = X.shape[0]
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (n // bs + 1))
    for ep in range(epochs):
        perm = th.randperm(n); net.train(); tot = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = th.nn.functional.cross_entropy(net(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(idx)
        net.eval()
        with th.no_grad():
            vl = sum(th.nn.functional.cross_entropy(net(Xv[i:i + 65536]), Yv[i:i + 65536], reduction='sum').item()
                     for i in range(0, Xv.shape[0], 65536)) / Xv.shape[0]
        print(f'epoch {ep} train {tot / n:.4f} val {vl:.4f}', flush=True)
        th.save(net.state_dict(), PATH)


if __name__ == '__main__':
    th.set_num_threads(2)
    if sys.argv[1] == 'train':
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 60, int(sys.argv[3]) if len(sys.argv) > 3 else 8)
