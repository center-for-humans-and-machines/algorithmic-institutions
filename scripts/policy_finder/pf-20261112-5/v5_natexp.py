"""v5: natural experiments from shared seeds. Same slot, same seat, same episode, two versions' points: the games are
identical until the first round r* where the punishment differs. Per such game pair, record r*, the punishment
difference at r* per player (B - A) with the player's context (contribution, newcomer, my group size), and the outcome
difference from r* on (lead, own, other pool summed over rounds r*..23, divided by 24; my members mean over r*+1..23).
Saves /tmp/pf/natexp.npy; columns: r*, slot, seat, episode, dOwn, dOther, dMembers, dp[8], c[8], mine[8], new[8],
my size, later points B-A, later points A+B."""
import numpy as np, itertools
V = ['v1', 'v2', 'v3', 'v4']
D = {}
for v in V:
    z = np.load(f'/tmp/pf/{v}.npz')
    D[v] = dict(c=z['c'], p=z['p'], m=z['mine'].astype(bool), val=z['v'], point=z['point'], seat=z['seat'])
    # within a (point, seat) block episodes are in order 0..E-1
recs = []
for a, b in itertools.combinations(V, 2):
    A, B = D[a], D[b]
    for k in np.unique(A['point']):
        for s in (0, 1):
            ia = np.where((A['point'] == k) & (A['seat'] == s))[0]; ib = np.where((B['point'] == k) & (B['seat'] == s))[0]
            if len(ia) != len(ib) or len(ia) == 0: continue
            cA, cB = A['c'][ia], B['c'][ib]; pA, pB = A['p'][ia].astype(np.int16), B['p'][ib].astype(np.int16)
            mA, mB = A['m'][ia], B['m'][ib]
            pdiff = (pA != pB).any(1)  # [E, 24]
            has = pdiff.any(1)
            rstar = np.where(has, pdiff.argmax(1), 24)
            # check identity of c and m up to rstar inclusive
            E = len(ia); rr = np.arange(24)
            upto = rr[None, :] <= rstar[:, None]
            same_c = ((cA == cB).all(1) | ~upto).all(1); same_m = ((mA == mB).all(1) | ~upto).all(1)
            ok = has & same_c & same_m
            if (~(same_c & same_m) & has).sum() > 0.01 * E:
                print('warn: history mismatch', a, b, k, s, (~(same_c & same_m)).sum())
            val = A['val'][ia]
            for side, (cc, pp, mm, vv) in enumerate([(cA, pA, mA, val), (cB, pB, mB, B['val'][ib])]):
                pool = 1.6 * cc * vv - pp
                own = (pool * mm).sum(1); oth = (pool * ~mm).sum(1)
                pm = (pp * mm).sum(1)  # my manager's points per round
                if side == 0: oA, tA, nA, qA = own, oth, mm.sum(1), pm
                else: oB, tB, nB, qB = own, oth, mm.sum(1), pm
            for e in np.where(ok)[0]:
                r = rstar[e]
                cum = lambda x: x[e, r:].sum() / 24.0
                dp = (pB[e, :, r] - pA[e, :, r]) * mA[e, :, r]
                new = (~mA[e, :, r - 1]) if r > 0 else np.zeros(8, bool)
                recs.append((r, k, s, e, cum(oB) - cum(oA), cum(tB) - cum(tA), (nB[e, r + 1:].mean() - nA[e, r + 1:].mean()) if r < 23 else 0.0,
                             *dp.tolist(), *cA[e, :, r].tolist(), *mA[e, :, r].astype(int).tolist(), *new.astype(int).tolist(), mA[e, :, r].sum(), qB[e, r + 1:].sum() - qA[e, r + 1:].sum(), qA[e, r + 1:].sum() + qB[e, r + 1:].sum()))
    print(a, b, len(recs), flush=True)
R = np.array(recs, dtype=np.float32)
np.save('/tmp/pf/natexp.npy', R)
print(R.shape)
