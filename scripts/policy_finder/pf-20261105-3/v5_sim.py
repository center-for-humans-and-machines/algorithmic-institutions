"""v5: v4's surrogate simulator, pointed at the refit v5 surrogate (/tmp/v5_surr.pt).
python v5_sim.py check [n]  -> surrogate vs sweep score on v4's 256 runs (+ a sample of
earlier runs) and on v4's six final points."""
import sys, json, warnings, numpy as np, torch as th
warnings.filterwarnings('ignore')
sys.path.insert(0, '/opt/algorithmic-institutions/scripts/policy_finder/pf-20261105-3')
import v4_sim, v3_load
v4_sim.net('/tmp/v5_surr.pt')
from v4_sim import simulate, rule_fn
ROOT = '/opt/algorithmic-institutions/'


def check(n=300):
    rows = []
    for ver, step in (('pf-20261105-3-v4', 1), ('pf-20261105-3-v2', 4), ('pf-20261105-3-v3', 4)):
        a = v3_load.load(ver)
        yml = f'configs/managers/rule_based/{ver}.yml'
        for name, prm, pool in list(zip(a['runs'], a['params'], a['pool']))[::step]:
            s, se = simulate(rule_fn(yml, json.loads(prm)), n=n, seed=1)
            rows.append((ver, name, pool, s))
    r = np.array([(x[2], x[3]) for x in rows])
    for ver in ('pf-20261105-3-v4', 'pf-20261105-3-v2', 'pf-20261105-3-v3'):
        m = np.array([x[0] == ver for x in rows])
        print(ver, 'n', m.sum(), 'pearson', round(np.corrcoef(r[m].T)[0, 1], 3), 'bias', round((r[m, 1] - r[m, 0]).mean(), 2),
              'rmse', round(np.sqrt(((r[m, 1] - r[m, 0]) ** 2).mean()), 2), flush=True)
    fin = json.load(open(ROOT + 'plots/simulation/policy_finder/pf-20261105-3-v4_sweep/final/sweep.json'))
    for pt in fin['points']:
        s = np.mean([simulate(rule_fn('configs/managers/rule_based/pf-20261105-3-v4.yml', pt['params']), n=4000, seed=k)[0] for k in (11, 12)])
        print('final', pt['name'], round(pt['pool'], 1), 'surrogate', round(s, 1), flush=True)


if __name__ == '__main__':
    th.set_num_threads(2)
    if sys.argv[1] == 'check':
        check(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
