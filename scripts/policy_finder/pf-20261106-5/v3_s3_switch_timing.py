# v3: does the timing of punishment within a block matter for switching (artificial humans)?
# v2 final data, all managers, both groups. One row per player x decision (after rounds 3,7,11,15,19).
# Arrays: games x 8 players x 24 rounds.
import pandas as pd, numpy as np, statsmodels.api as sm, pyarrow.parquet as pq, sys
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
fn = sys.argv[1] if len(sys.argv) > 1 else base + 'pf-20261106-5-v2_sweep/final/per_round.parquet'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
out = []
tb = pq.read_table(fn, columns=cols, read_dictionary=['run'])
d = tb.to_pandas()
del tb
d['run'] = d.run.cat.codes
for i in [0]:
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number'])
    n = len(d) // 192
    P = d.punishment.values.reshape(n, 8, 24).astype(float)
    C = d.contribution.values.reshape(n, 8, 24).astype(float)
    G = d.group_id.values.reshape(n, 8, 24).astype(int)
    for dec in [3, 7, 11, 15, 19]:
        rr = slice(dec - 3, dec + 1)
        g = G[:, :, dec]
        sw = (G[:, :, dec + 1] != g).astype(int)
        own = (G[:, :, rr] == g[:, :, None])  # same group as me in that round
        oth = ~own
        def gmean(X, M):
            # mean over players of X per round where in mask relative to each player
            s = np.einsum('npr,nqr->nqr', X, np.ones_like(X))  # placeholder
            return s
        Pb, Cb, Gb = P[:, :, rr], C[:, :, rr], G[:, :, rr]
        res = {}
        for j in range(4):
            gj = Gb[:, :, j]  # n x 8
            same = (gj[:, :, None] == gj[:, None, :])  # n x 8 x 8 (whether q in same group as p at round j)
            # own group: group of the deciding player at round j
            pj = Pb[:, :, j]; cj = Cb[:, :, j]
            ns = same.sum(2); no = 8 - ns
            res[f'own_gp{j}'] = (same * pj[:, None, :]).sum(2) / ns
            res[f'oth_gp{j}'] = np.where(no > 0, ((~same) * pj[:, None, :]).sum(2) / np.maximum(no, 1), 0)
            res[f'p{j}'] = pj
        gd = G[:, :, dec]; same = gd[:, :, None] == gd[:, None, :]
        cm = Cb.mean(2)
        x = pd.DataFrame({k: v.ravel() for k, v in res.items()})
        x['sw'] = sw.ravel(); x['dec'] = dec; x['c_mean'] = cm.ravel()
        x['own_n'] = same.sum(2).ravel()
        x['own_gc'] = ((same * C[:, None, :, dec]).sum(2) / same.sum(2)).ravel()
        x['oth_gc'] = np.where(~same.all(2), ((~same) * C[:, None, :, dec]).sum(2) / np.maximum((~same).sum(2), 1), 0).ravel()
        out.append(x)
X = pd.concat(out, ignore_index=True)
print(len(X), X.sw.mean().round(3))
feats = ['p0', 'p1', 'p2', 'p3', 'own_gp0', 'own_gp1', 'own_gp2', 'own_gp3', 'oth_gp0', 'oth_gp1', 'oth_gp2', 'oth_gp3', 'c_mean', 'own_gc', 'oth_gc', 'own_n']
m = sm.Logit(X.sw, sm.add_constant(X[feats].astype(float))).fit(disp=0)
print(pd.DataFrame({'coef': m.params, 'se': m.bse}).round(3).to_string())
X['pe'] = pd.cut(X.own_gp0 + X.own_gp1, [-1, 0.01, 4, 10, 61]); X['pl'] = pd.cut(X.own_gp2 + X.own_gp3, [-1, 0.01, 4, 10, 61])
print(X.pivot_table(index='pe', columns='pl', values='sw', aggfunc=['mean', 'size'], observed=True).round(3))
X.drop(columns=["pe", "pl"]).to_parquet("/tmp/v3_switch.parquet")
