"""v6: cache v4's and v5's sweeps as npz (cache() copied from v4_cache_dyn)."""
import pandas as pd, numpy as np, json, os
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

cache('pf-20261105-8-v4', ('standard', 'rate', 'dose', 'opening', 'horizon', 'floor', 'end'), '/tmp/v4sweep.npz')
cache('pf-20261105-8-v5', ('standard', 'rate', 'dose', 'horizon', 'end', 'floor'), '/tmp/v5sweep.npz')
