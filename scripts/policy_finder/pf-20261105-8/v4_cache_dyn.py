"""v4: cache all three sweeps to /tmp npz (as v3_cache), and print dynamics of the finals' points."""
import pandas as pd, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
def cache(version, keys, out):
    if os.path.exists(out): return
    base = f'plots/simulation/policy_finder/{version}_sweep/'
    sw = json.load(open(base + 'sweep.json'))
    idx = {p['name']: k for k, p in enumerate(sw['points'])}
    Cs, Ps, Vs, Rs = [], [], [], []
    for part in ('per_round_p1of2.parquet', 'per_round_p2of2.parquet'):
        d = pd.read_parquet(base + part, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'run'])
        d['player'] = d.participant_code.str.split('_').str[0].astype(int)
        d['ri'] = d.run.str.split(' managed by ').str[1].map(idx)
        d = d.sort_values(['ri', 'episode', 'player', 'round_number'])
        n = len(d) // 96
        Cs.append(d.contribution.values.reshape(n, 4, 24)); Ps.append(d.punishment.values.reshape(n, 4, 24))
        Vs.append(d.contribution_valid.values.reshape(n, 4, 24).astype(bool)); Rs.append(d.ri.values.reshape(n, 96)[:, 0])
        del d
    C, P, V, R = (np.concatenate(x) for x in (Cs, Ps, Vs, Rs))
    prm = np.array([[p['params'][k] for k in keys] for p in sw['points']])
    np.savez_compressed(out, C=C.astype(np.int8), P=P.astype(np.int8), V=V, R=R, prm=prm,
                        pool=np.array([p['pool'] for p in sw['points']]))
    print(version, C.shape)
cache('pf-20261105-8', ('target', 'slope', 'horizon', 'cap'), '/tmp/v1sweep.npz')
cache('pf-20261105-8-v2', ('standard', 'dose', 'curvature', 'horizon'), '/tmp/v2sweep.npz')
cache('pf-20261105-8-v3', ('standard', 'dose', 'curvature', 'horizon', 'floor', 'tail'), '/tmp/v3sweep.npz')

def summ(d):
    v = d[d.contribution_valid.astype(bool)]
    g = v.groupby('round_number')
    return pd.DataFrame({'c': g.contribution.mean(), 'p': g.punishment.mean(), 'pay': g.payoff.mean(),
                         'c<5': g.contribution.apply(lambda x: (x < 5).mean()),
                         'pun': g.punishment.apply(lambda x: (x > 0).mean())})
for v in ['pf-20261105-8-v2', 'pf-20261105-8-v3']:
    f = pd.read_parquet(f'plots/simulation/policy_finder/{v}_sweep/final/per_round.parquet')
    for run, d in f.groupby('run'):
        s = summ(d)
        vv = d[d.contribution_valid.astype(bool)]
        sc = vv.groupby(['episode', 'round_number']).payoff.mean().groupby('episode').sum().mean()
        print(run.split(' by ')[1], round(sc, 1), 'totP/player', round(s.p.sum(), 1))
        print(s.round(1).T.to_string())
