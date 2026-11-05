"""v2: empirical one-step surrogate of the clones built from v1's sweep (128k games).
Next contribution is drawn from sweep rows in the same cell of
(c bin, c(t-1) bin, own mean c so far bin, others' mean bin, p bin, number of earlier punishments bin, round bin),
falling back to coarser cells when a cell has < MIN rows. Timeouts ignored.
Validated by replaying v1's 256 sweep rules and comparing with their true scores."""
import numpy as np, json
CB = np.array([0, 1, 4, 7, 10, 12, 14, 16, 19, 20.5])     # c bins edges
OB = np.array([0, 4, 7, 9, 11, 13, 15, 17, 20.5])
PB = np.array([0, 0.5, 2.5, 5.5, 9.5, 14.5, 20.5, 26.5, 30.5])
QB = np.array([0, 0.5, 2, 4, 7, 11, 30.5])  # others' mean punishment
NB = np.array([0, 0.5, 1.5, 3.5, 30])
HB = np.array([0, 3, 6, 9, 12, 15, 18, 20.5])
TB = np.array([0, 0.5, 1.5, 3.5, 6.5, 10.5, 15.5, 24])
MIN = 30
def dig(x, e): return np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2)
dims = [len(CB) - 1, len(CB) - 1, len(OB) - 1, len(PB) - 1, len(NB) - 1, len(TB) - 1, len(HB) - 1, len(QB) - 1]
def keys(c, cp, o, p, n, t, h, q, level):
    k = [dig(c, CB), dig(cp, CB), dig(o, OB), dig(p, PB), dig(n, NB), dig(t, TB), dig(h, HB), dig(q, QB)]
    if level >= 1: k[4] = 0 * k[4]                    # drop npun
    if level >= 2: k[1] = 0 * k[1]                    # drop cprev
    if level >= 3: k[6] = 0 * k[6]                    # drop own history mean
    if level >= 4: k[5] = 0 * k[5]; k[7] = 0 * k[7]   # drop t, others' p
    return np.ravel_multi_index(k, dims)
class Surrogate:
    def __init__(self, path='/tmp/v1sweep.npz', sub=1):
        z = np.load(path)
        C, P, V = z['C'][::sub].astype(np.float32), z['P'][::sub].astype(np.float32), z['V'][::sub]
        G = C.shape[0]
        npun = np.concatenate([np.zeros((G, 4, 1)), np.cumsum(P > 0, 2)[:, :, :-1]], 2)
        cs, cps, os_, ps, ns, ts, ys, hs, qs = [], [], [], [], [], [], [], [], []
        hm = np.cumsum(C, 2) / np.arange(1, 25)
        for t in range(0, 23):
            for i in range(4):
                o = [j for j in range(4) if j != i]
                m = V[:, i, t] & V[:, i, t + 1] & V[:, :, t].all(1)
                g = np.where(m)[0]
                cs.append(C[g, i, t]); cps.append(C[g, i, t - 1] if t > 0 else C[g, i, t])
                os_.append(C[g][:, o, t].mean(1)); ps.append(P[g, i, t]); ns.append(npun[g, i, t])
                ts.append(np.full(len(g), t)); hs.append(hm[g, i, t]); qs.append(P[g][:, o, t].mean(1)); ys.append(C[g, i, t + 1])
        self.cols = [np.concatenate(x) for x in (cs, cps, os_, ps, ns, ts, hs, qs)]
        self.y = np.concatenate(ys).astype(np.int8)
        self.tables = []
        for lev in range(5):
            print('table', lev, flush=True)
            k = keys(*self.cols, lev)
            order = np.argsort(k, kind='stable')
            ks = k[order]
            size = int(np.prod(dims))
            cnt = np.bincount(ks, minlength=size); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            self.tables.append((self.y[order], start, cnt))
        r0 = V[:, :, 0].all(1)
        self.c0 = C[r0][:, :, 0]
    def step(self, c, cp, o, p, n, t, h, rng):
        q = (p.sum(1, keepdims=True) - p) / 3
        out = np.full(c.shape, -1.0)
        todo = np.ones(c.shape, bool)
        for lev, (ys, start, cnt) in enumerate(self.tables):
            k = keys(c, cp, o, p, n, np.full(c.shape, t), h, q, lev)
            ok = todo & ((cnt[k] >= MIN) | (lev == 4))
            idx = start[k[ok]] + np.floor(rng.random(ok.sum()) * np.maximum(cnt[k[ok]], 1)).astype(int)
            out[ok] = ys[np.minimum(idx, len(ys) - 1)]
            todo &= ~ok
        return out
    def play(self, rule, n=4000, seed=0):
        """rule(C_hist [n,4,t+1], P_hist [n,4,t], t) -> punishment [n,4]. Returns score and per-round means."""
        rng = np.random.default_rng(seed)
        Ch = np.zeros((n, 4, 24)); Ph = np.zeros((n, 4, 24))
        Ch[:, :, 0] = self.c0[rng.integers(0, len(self.c0), n)]
        score = np.zeros(n)
        for t in range(24):
            c = Ch[:, :, t]
            p = np.floor(np.clip(rule(Ch[:, :, :t + 1], Ph[:, :, :t], t), 0, 30))
            Ph[:, :, t] = p
            score += (20 - c - p + (1.6 * c.sum(1, keepdims=True) - p.sum(1, keepdims=True)) / 4).mean(1)
            if t < 23:
                cp = Ch[:, :, t - 1] if t > 0 else c
                o = (c.sum(1, keepdims=True) - c) / 3
                npn = (Ph[:, :, :t] > 0).sum(2)
                h = Ch[:, :, :t + 1].mean(2)
                Ch[:, :, t + 1] = self.step(c, cp, o, p, npn, t, h, rng)
        return score.mean(), score.std() / np.sqrt(n), Ch.mean((0, 1)), Ph.mean((0, 1))
def v1_rule(target, slope, horizon, cap):
    def r(Ch, Ph, t):
        if t >= horizon: return np.zeros(Ch.shape[:2])
        return np.minimum(slope * np.clip(target - Ch[:, :, -1], 0, None), cap)
    return r
if __name__ == '__main__':
    S = Surrogate()
    sw = json.load(open('plots/simulation/policy_finder/pf-20261105-8_sweep/sweep.json'))
    true, est = [], []
    for k, pt in enumerate(sw['points']):
        pr = pt['params']
        s = S.play(v1_rule(pr['target'], pr['slope'], pr['horizon'], pr['cap']), n=1500, seed=k)[0]
        true.append(pt['pool']); est.append(s)
    true, est = np.array(true), np.array(est)
    print('corr', np.corrcoef(true, est)[0, 1].round(3), 'mean bias', (est - true).mean().round(1), 'rmse', np.sqrt(((est - true) ** 2).mean()).round(1))
    top = np.argsort(-true)[:30]
    print('corr top30', np.corrcoef(true[top], est[top])[0, 1].round(3))
    print('zero rule', S.play(lambda Ch, Ph, t: np.zeros(Ch.shape[:2]), n=4000)[0])
    print('v1 final', S.play(v1_rule(13.75, 2.23, 12, 30), n=8000)[:2])
    np.save('/tmp/surr_check.npy', np.stack([true, est]))
